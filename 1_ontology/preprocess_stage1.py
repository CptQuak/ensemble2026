from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd
from rdkit import Chem


CLASS_RE = re.compile(r"^class_(\d+)$")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use .csv or .parquet.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {suffix}. Use .csv or .parquet.")


def find_class_columns(columns: Iterable[str]) -> List[str]:
    pairs = []
    for col in columns:
        m = CLASS_RE.match(col)
        if m:
            pairs.append((int(m.group(1)), col))
    pairs.sort(key=lambda x: x[0])
    return [col for _, col in pairs]


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    normalized = series.fillna("false").astype(str).str.strip().str.lower()
    return normalized.map(mapping).fillna(False).astype(bool)


def parse_obo_parents(obo_path: Path) -> Dict[str, Set[str]]:
    parents: Dict[str, Set[str]] = {}
    current_id: Optional[str] = None

    with obo_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "[Term]":
                current_id = None
                continue
            if line.startswith("id: "):
                current_id = line[4:].strip()
                parents.setdefault(current_id, set())
                continue
            if line.startswith("is_a: ") and current_id:
                parent_id = line[6:].split("!")[0].strip()
                if parent_id:
                    parents[current_id].add(parent_id)
    return parents


def compute_ancestors_map(
    class_cols: List[str], parent_map: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    allowed = set(class_cols)
    memo: Dict[str, Set[str]] = {}

    def dfs(node: str, stack: Optional[Set[str]] = None) -> Set[str]:
        if node in memo:
            return memo[node]
        if stack is None:
            stack = set()
        if node in stack:
            return set()
        stack = set(stack)
        stack.add(node)

        direct = {p for p in parent_map.get(node, set()) if p in allowed}
        out: Set[str] = set(direct)
        for p in direct:
            out.update(dfs(p, stack))
        memo[node] = out
        return out

    for cls in class_cols:
        dfs(cls)
    return memo


def count_hierarchy_violations(
    df: pd.DataFrame, class_cols: List[str], ancestors_map: Dict[str, Set[str]]
) -> int:
    violations = 0
    for child in class_cols:
        ancestors = ancestors_map.get(child, set())
        if not ancestors:
            continue
        child_true = df[child]
        for parent in ancestors:
            violations += int((child_true & (~df[parent])).sum())
    return violations


def apply_hierarchy_closure(
    df: pd.DataFrame, class_cols: List[str], ancestors_map: Dict[str, Set[str]]
) -> int:
    updates = 0
    for child in class_cols:
        ancestors = ancestors_map.get(child, set())
        if not ancestors:
            continue
        child_true = df[child]
        if not bool(child_true.any()):
            continue
        for parent in ancestors:
            mask = child_true & (~df[parent])
            n = int(mask.sum())
            if n > 0:
                df.loc[mask, parent] = True
                updates += n
    return updates


def safe_mol_from_smiles(smiles: str):
    if not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 preprocessing for hierarchical molecular classification."
    )
    parser.add_argument("--input", required=True, help="Input CSV/Parquet path.")
    parser.add_argument("--output", required=True, help="Output CSV/Parquet path.")
    parser.add_argument(
        "--obo",
        default="1_ontology/extras/chebi_classes.obo",
        help="Path to OBO hierarchy file.",
    )
    parser.add_argument(
        "--definitions",
        default="1_ontology/extras/chebi_class_definitions.csv",
        help="Optional class definitions CSV for reporting class names.",
    )
    parser.add_argument(
        "--drop-invalid-smiles",
        action="store_true",
        help="Drop rows with invalid SMILES.",
    )
    parser.add_argument(
        "--deduplicate-by",
        choices=["none", "canonical_smiles", "inchikey"],
        default="inchikey",
        help="Deduplication key.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional report path (default: <output>.report.json).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    obo_path = Path(args.obo)
    defs_path = Path(args.definitions)
    report_path = (
        Path(args.report_json) if args.report_json else output_path.with_suffix(".report.json")
    )

    df = read_table(input_path)
    if "mol_id" not in df.columns or "SMILES" not in df.columns:
        raise ValueError("Input must contain 'mol_id' and 'SMILES' columns.")

    class_cols = find_class_columns(df.columns)
    if not class_cols:
        raise ValueError("No class columns found (expected columns like class_0 ... class_499).")

    for col in class_cols:
        df[col] = to_bool_series(df[col])

    n_rows_initial = len(df)

    df["mol"] = df["SMILES"].apply(safe_mol_from_smiles)
    df["is_valid_smiles"] = df["mol"].notna()

    # Canonical strings/keys are computed from parsed RDKit molecules.
    df["canonical_smiles"] = df["mol"].apply(
        lambda m: Chem.MolToSmiles(m, canonical=True) if m is not None else None
    )
    df["inchikey"] = df["mol"].apply(
        lambda m: Chem.MolToInchiKey(m) if m is not None else None
    )

    n_invalid_smiles = int((~df["is_valid_smiles"]).sum())
    if args.drop_invalid_smiles:
        df = df[df["is_valid_smiles"]].copy()

    n_after_invalid = len(df)
    duplicates_removed = 0
    if args.deduplicate_by != "none":
        key = args.deduplicate_by
        before = len(df)
        if key == "inchikey":
            df = df.drop_duplicates(subset=["inchikey"], keep="first")
        else:
            df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
        duplicates_removed = before - len(df)

    parent_map = parse_obo_parents(obo_path)
    ancestors_map = compute_ancestors_map(class_cols, parent_map)

    violations_before = count_hierarchy_violations(df, class_cols, ancestors_map)
    closure_updates = apply_hierarchy_closure(df, class_cols, ancestors_map)
    violations_after = count_hierarchy_violations(df, class_cols, ancestors_map)

    class_name_map: Dict[str, str] = {}
    if defs_path.exists():
        defs_df = pd.read_csv(defs_path)
        if {"chebi_id", "name"}.issubset(defs_df.columns):
            class_name_map = dict(zip(defs_df["chebi_id"], defs_df["name"]))

    prevalence = df[class_cols].mean().sort_values(ascending=False)
    top5 = []
    for cls, frac in prevalence.head(5).items():
        top5.append(
            {
                "class_id": cls,
                "class_name": class_name_map.get(cls, ""),
                "positive_fraction": float(frac),
            }
        )

    df = df.drop(columns=["mol"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(df, output_path)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "n_rows_input": n_rows_initial,
        "n_class_columns": len(class_cols),
        "n_invalid_smiles": n_invalid_smiles,
        "drop_invalid_smiles": args.drop_invalid_smiles,
        "n_rows_after_invalid_filter": n_after_invalid,
        "deduplicate_by": args.deduplicate_by,
        "duplicates_removed": duplicates_removed,
        "n_rows_output": len(df),
        "hierarchy_violations_before": violations_before,
        "hierarchy_updates_applied": closure_updates,
        "hierarchy_violations_after": violations_after,
        "top5_most_frequent_classes": top5,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Saved report to: {report_path}")
    print(
        "Summary: "
        f"input={n_rows_initial}, output={len(df)}, "
        f"invalid_smiles={n_invalid_smiles}, duplicates_removed={duplicates_removed}, "
        f"violations_before={violations_before}, violations_after={violations_after}"
    )


if __name__ == "__main__":
    main()
