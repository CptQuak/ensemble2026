"""
chebi_data.py
=============
Ładowanie i podział zbioru danych ChEBI.

Eksportuje:
    load_chebi_dataset   – wczytuje parquet do DataFrame
    scaffold_split       – podział train/valid oparty na rusztowaniach Bemis-Murcko
    get_label_columns    – pomocnik zwracający kolumny etykiet z prefixem
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ładowanie danych
# ---------------------------------------------------------------------------

def load_chebi_dataset(data_path: str) -> pd.DataFrame:
    """Wczytuje zbiór danych ChEBI z pliku parquet.

    Args:
        data_path: Ścieżka do pliku .parquet.

    Returns:
        DataFrame z SMILES i kolumnami etykiet.
    """
    df = pd.read_parquet(data_path)
    log.info(f"Wczytano dane: {df.shape[0]} próbek, {df.shape[1]} kolumn.")
    return df


def get_label_columns(df: pd.DataFrame, prefix: str = "class_") -> List[str]:
    """Zwraca listę kolumn etykiet zaczynających się od podanego prefixu.

    Args:
        df:     DataFrame z kolumnami etykiet.
        prefix: Prefix identyfikujący kolumny etykiet (domyślnie 'class_').

    Returns:
        Posortowana lista nazw kolumn etykiet.
    """
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    log.info(f"Znaleziono {len(cols)} kolumn etykiet (prefix='{prefix}').")
    return cols


# ---------------------------------------------------------------------------
# Scaffold Split
# ---------------------------------------------------------------------------

def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Generuje rusztowanie Bemis-Murcko dla podanego ciągu SMILES.

    Args:
        smiles:            Ciąg znaków SMILES molekuły.
        include_chirality: Czy uwzględniać stereochemię w rusztowaniu.

    Returns:
        SMILES rusztowania lub pusty ciąg, jeśli parsowanie zawiodło.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    train_size: float = 0.8,
    include_chirality: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wykonuje podział zbioru danych w oparciu o unikalne rusztowania molekularne.

    Gwarantuje, że wszystkie molekuły o tym samym rdzeniu (scaffold) trafiają
    wyłącznie do jednego ze zbiorów – zapobiega wyciekowi strukturalnemu.

    Grupy sortowane są malejąco wg rozmiaru: największe scaffoldy trafiają do
    train, pozostałe do valid (zachowanie standardowe dla ChEBI).

    Args:
        df:                Wejściowy zbiór danych ChEBI.
        smiles_col:        Nazwa kolumny z ciągami SMILES.
        train_size:        Proporcja zbioru treningowego (domyślnie 0.8).
        include_chirality: Czy uwzględniać stereochemię w generowaniu scaffoldów.

    Returns:
        Tuple (df_train, df_valid).
    """
    log.info("Generowanie rusztowań Bemis-Murcko dla scaffold split...")

    # Grupowanie indeksów wg unikalnych rusztowań
    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for idx, smiles in enumerate(df[smiles_col]):
        scaffold_smi = _generate_scaffold(smiles, include_chirality)
        scaffolds[scaffold_smi].append(idx)

    # Sortowanie od największej do najmniejszej grupy (stabilizuje podział)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_indices: List[int] = []
    valid_indices: List[int] = []
    n_train_cutoff = int(len(df) * train_size)

    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= n_train_cutoff:
            train_indices.extend(scaffold_set)
        else:
            valid_indices.extend(scaffold_set)

    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_valid  = df.iloc[valid_indices].reset_index(drop=True)

    log.info(
        f"Scaffold split zakończony: train={len(df_train)}, valid={len(df_valid)} "
        f"(proporcja docelowa: {train_size:.0%})"
    )
    return df_train, df_valid
