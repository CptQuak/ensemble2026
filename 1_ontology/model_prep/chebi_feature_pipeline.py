"""
chebi_feature_pipeline.py
=========================
Moduł inżynierii cech dla klasyfikacji ontologii ChEBI.

Zawiera rozszerzony potok cech z możliwością włączania/wyłączania
poszczególnych grup cech poprzez flagi konfiguracyjne.

Nowe grupy cech (v2) targetowane na słabo klasyfikowane klasy:
    - SaltIonFeatureExtractor      → sole/jony (zwitterion, hydrochloride, ammonium)
    - RingTopologyFeatureExtractor → złożone pierścienie (tetracyclic, spiro, macrolide)
    - SmartsPatternFeatureExtractor → heterocykle (pyrazole, triazole, benzofuran...)
    - ChainFeatureExtractor        → kwasy tłuszczowe i związki alifatyczne
    - PolymerFeatureExtractor      → biomakrocząsteczki i oligosacharydy

Eksportuje:
    FeatureConfig            – dataclass z flagami grup cech
    ChEBIFeaturePipeline     – klasa zarządzająca osobnymi ścieżkami train/valid
    build_chebi_feature_pipeline – niskopoziomowy builder potoku sklearn
    preprocess_chebi_data    – wygodna funkcja dla szybkiego użycia
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops
from rdkit.Chem.SaltRemover import SaltRemover
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from skfp.fingerprints import (
    MACCSFingerprint,
    ECFPFingerprint,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    KlekotaRothFingerprint,
    RDKit2DDescriptorsFingerprint,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konfiguracja flag cech
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Konfiguracja włączania/wyłączania poszczególnych grup cech.

    Grupy oryginalne:
        use_maccs:              MACCS Keys – 166 predefinowanych grup funkcyjnych.
        use_ecfp_count:         ECFP Count – lokalne otoczenia promieniowe (hashed).
        use_atom_pair:          Atom Pair – odległości topologiczne między parami atomów.
        use_topo_torsion:       Topological Torsion – czwórki atomów, kąty skręcenia.
        use_klekota_roth:       Klekota-Roth – 4860 wzorców farmakoforycznych.
        use_rdkit_descs:        RDKit 2D Descriptors – ~200 właściwości fizykochemicznych.
        use_domain_features:    Ręcznie wyznaczone cechy domenowe (pierścienie, ładunek itp.).
        use_smiles_features:    Cechy ze stringa SMILES (sole, chiralność, jony itp.).

    Grupy nowe (v2) – targetowane na słabo klasyfikowane klasy:
        use_salt_ion_features:  Cechy soli/jonów z cząsteczki PRZED usunięciem soli.
                                Poprawia: zwitterion, hydrochloride, ammonium compound,
                                organoammonium salt, chloride salt, halide salt.
        use_ring_topology:      Globalna topologia pierścieni (makrocykle, spiro, fused).
                                Poprawia: tetracyclic compound, macrolide, spiro compound,
                                oxaspiro compound, triterpenoid.
        use_smarts_patterns:    Wzorce SMARTS bezpośrednio mapujące definicje ChEBI.
                                Poprawia: pyrazoles, triazoles, benzofurans, aminopyrimidine,
                                piperazines, chromenes, anilines i inne heterocykle.
        use_chain_features:     Długość i liniowość łańcucha węglowego.
                                Poprawia: fatty acid, aliphatic compound, alicyclic compound,
                                olefin, cyclic olefin.
        use_polymer_features:   Motywy polimeryczne i sacharydowe.
                                Poprawia: biomacromolecule, trisaccharide derivative,
                                oligosaccharide derivative.
    """
    # --- Oryginalne flagi ---
    use_maccs: bool = True
    use_ecfp_count: bool = True
    use_atom_pair: bool = True
    use_topo_torsion: bool = True
    use_klekota_roth: bool = True
    use_rdkit_descs: bool = True
    use_domain_features: bool = True
    use_smiles_features: bool = True

    # --- Nowe flagi v2 ---
    use_salt_ion_features: bool = True
    use_ring_topology: bool = True
    use_smarts_patterns: bool = True
    use_chain_features: bool = True
    use_polymer_features: bool = True

    # --- Hiperparametry fingerprints ---
    ecfp_radius: int = 2
    ecfp_n_bits: int = 2048
    atom_pair_count: bool = True
    topo_torsion_count: bool = True

    # --- Parametry obliczeniowe ---
    n_jobs: int = -1


# ---------------------------------------------------------------------------
# Standardyzator molekularny — zwraca KROTKĘ (oryginał, czysty)
# ---------------------------------------------------------------------------

class DAGMolStandardizer(BaseEstimator, TransformerMixin):
    """Parsuje SMILES i zwraca krotkę (mol_original, mol_stripped).

    mol_original: cząsteczka przed usunięciem soli — potrzebna dla
                  SaltIonFeatureExtractor (informacja o jonach/fragmentach).
    mol_stripped:  cząsteczka po usunięciu soli — używana przez wszystkie
                  pozostałe ekstraktory i fingerprints skfp.

    Nie filtruje dużych molekuł — ontologia ChEBI zawiera makromolekuły.
    """

    def __init__(self) -> None:
        self.salt_remover = SaltRemover()

    def fit(self, X: List[str], y: Optional[Any] = None) -> "DAGMolStandardizer":
        return self

    def transform(self, X: List[str]) -> List[Tuple[Optional[Chem.Mol], Optional[Chem.Mol]]]:
        result = []
        for smiles in X:
            mol_orig = Chem.MolFromSmiles(smiles)
            if mol_orig is None:
                log.warning(f"Odrzucono uszkodzony SMILES: {smiles}")
                result.append((None, None))
                continue
            mol_clean = self.salt_remover.StripMol(mol_orig, dontRemoveEverything=True)
            if mol_clean is None:
                mol_clean = mol_orig
            try:
                Chem.SanitizeMol(mol_clean)
            except Exception as e:
                log.warning(f"Błąd sanityzacji: {e}. Używam oryginału.")
                mol_clean = mol_orig
            result.append((mol_orig, mol_clean))
        return result


# ---------------------------------------------------------------------------
# Adaptery
# ---------------------------------------------------------------------------

class OriginalMolExtractor(BaseEstimator, TransformerMixin):
    """Wyciąga mol_original z krotek (original, stripped)."""
    def fit(self, X, y=None): return self
    def transform(self, X): return [pair[0] for pair in X]


class MolToSmilesTransformer(BaseEstimator, TransformerMixin):
    """Konwertuje krotki lub Mol na SMILES (stripped)."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        result = []
        for item in X:
            mol = item[1] if isinstance(item, tuple) else item
            result.append(Chem.MolToSmiles(mol) or "" if mol is not None else "")
        return result


class ScaledDenseToSparse(BaseEstimator, TransformerMixin):
    """Imputuje NaN/Inf, normalizuje (StandardScaler) i konwertuje do sparse CSR."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

    def _clean(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float32, copy=True)
        X[np.isinf(X)] = np.nan
        return X

    def fit(self, X, y=None):
        X_clean = self._clean(X)
        self.scaler.fit(self.imputer.fit_transform(X_clean))
        return self

    def transform(self, X) -> csr_matrix:
        X_clean = self._clean(X)
        return csr_matrix(self.scaler.transform(self.imputer.transform(X_clean)))


class SkfpMolAdapter(BaseEstimator, TransformerMixin):
    """Opakowuje fingerprinty skfp tak, żeby akceptowały krotki (orig, stripped)."""

    def __init__(self, fingerprint) -> None:
        self.fingerprint = fingerprint

    def fit(self, X, y=None):
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        self.fingerprint.fit(mols, y)
        return self

    def transform(self, X):
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        return self.fingerprint.transform(mols)

    def fit_transform(self, X, y=None):
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        return self.fingerprint.fit_transform(mols, y)


# ---------------------------------------------------------------------------
# Oryginalne ekstraktory domenowe
# ---------------------------------------------------------------------------

class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wyznacza ręcznie zaprojektowane cechy domenowe z grafów molekularnych."""

    _HETEROATOMS = ["N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "Se", "B"]

    def fit(self, X, y=None): return self

    def transform(self, X) -> np.ndarray:
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        return np.array([self._featurize(mol) for mol in mols], dtype=np.float32)

    def _featurize(self, mol):
        if mol is None:
            return [0.0] * self._n_features()

        feats = []
        feats.append(float(Chem.GetFormalCharge(mol)))
        feats.append(float(abs(Chem.GetFormalCharge(mol))))
        feats.append(float(sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())))
        feats.append(float(Descriptors.NumRadicalElectrons(mol)))

        ring_info = mol.GetRingInfo()
        feats.append(float(rdMolDescriptors.CalcNumRings(mol)))
        feats.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)))
        feats.append(float(rdMolDescriptors.CalcNumSaturatedRings(mol)))
        feats.append(float(rdMolDescriptors.CalcNumAliphaticRings(mol)))
        feats.append(float(rdMolDescriptors.CalcNumHeterocycles(mol)))
        feats.append(float(rdMolDescriptors.CalcNumAromaticHeterocycles(mol)))
        feats.append(float(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)))
        feats.append(float(rdMolDescriptors.CalcNumSpiroAtoms(mol)))
        feats.append(float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)))

        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        for size in range(3, 9):
            feats.append(float(sum(1 for s in ring_sizes if s == size)))
        feats.append(float(sum(1 for s in ring_sizes if s >= 9)))

        feats.append(float(rdMolDescriptors.CalcFractionCSP3(mol)))
        n_atoms = mol.GetNumAtoms()
        if n_atoms > 0:
            sp3 = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
            sp2 = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2)
            sp  = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.rdchem.HybridizationType.SP)
            feats.extend([sp3 / n_atoms, sp2 / n_atoms, sp / n_atoms])
        else:
            feats.extend([0.0, 0.0, 0.0])

        feats.append(float(rdMolDescriptors.CalcNumAtomStereoCenters(mol)))
        feats.append(float(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)))
        stereo_bonds = sum(
            1 for b in mol.GetBonds()
            if b.GetStereo() not in (Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY)
        )
        feats.append(float(stereo_bonds))

        feats.append(float(Descriptors.MolWt(mol)))
        feats.append(float(Descriptors.ExactMolWt(mol)))
        feats.append(float(Descriptors.TPSA(mol)))
        feats.append(float(Descriptors.MolLogP(mol)))
        feats.append(float(Descriptors.NumHDonors(mol)))
        feats.append(float(Descriptors.NumHAcceptors(mol)))
        feats.append(float(Descriptors.NumRotatableBonds(mol)))
        feats.append(float(Descriptors.RingCount(mol)))
        feats.append(float(Descriptors.HeavyAtomCount(mol)))
        feats.append(float(Descriptors.NumValenceElectrons(mol)))
        feats.append(float(Descriptors.MaxPartialCharge(mol) or 0.0))
        feats.append(float(Descriptors.MinPartialCharge(mol) or 0.0))

        bonds = list(mol.GetBonds())
        n_single = sum(1 for b in bonds if b.GetBondTypeAsDouble() == 1.0)
        n_double = sum(1 for b in bonds if b.GetBondTypeAsDouble() == 2.0)
        n_triple = sum(1 for b in bonds if b.GetBondTypeAsDouble() == 3.0)
        n_arom   = sum(1 for b in bonds if b.GetIsAromatic())
        n_bonds  = mol.GetNumBonds()
        feats.extend([float(n_single), float(n_double), float(n_triple), float(n_arom)])
        feats.append(float(n_arom) / n_bonds if n_bonds > 0 else 0.0)

        atom_counts = {sym: 0 for sym in self._HETEROATOMS}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in atom_counts:
                atom_counts[sym] += 1
        feats.extend([float(atom_counts[sym]) for sym in self._HETEROATOMS])

        n_heavy = max(mol.GetNumHeavyAtoms(), 1)
        for sym in self._HETEROATOMS:
            feats.append(atom_counts[sym] / n_heavy)

        frags = Chem.GetMolFrags(mol)
        feats.append(float(len(frags)))
        feats.append(float(Descriptors.BertzCT(mol)))
        feats.append(float(rdMolDescriptors.CalcLabuteASA(mol)))

        return feats

    def _n_features(self) -> int:
        return len(self._featurize(Chem.MolFromSmiles("C")))

    def get_feature_names_out(self, input_features=None):
        names = [
            "formal_charge", "abs_formal_charge", "sum_abs_atom_charges",
            "n_radical_electrons",
            "n_rings", "n_aromatic_rings", "n_saturated_rings",
            "n_aliphatic_rings", "n_heterocycles",
            "n_aromatic_heterocycles", "n_saturated_heterocycles",
            "n_spiro_atoms", "n_bridgehead_atoms",
            "n_ring_size_3", "n_ring_size_4", "n_ring_size_5",
            "n_ring_size_6", "n_ring_size_7", "n_ring_size_8", "n_ring_size_9plus",
            "frac_csp3", "frac_sp3", "frac_sp2", "frac_sp",
            "n_stereocenters", "n_unspec_stereocenters", "n_stereo_bonds",
            "mol_wt", "exact_mol_wt", "tpsa", "logp",
            "n_hdonors", "n_hacceptors", "n_rotatable_bonds",
            "ring_count", "heavy_atom_count", "n_valence_electrons",
            "max_partial_charge", "min_partial_charge",
            "n_single_bonds", "n_double_bonds", "n_triple_bonds",
            "n_aromatic_bonds", "frac_aromatic_bonds",
        ]
        names += [f"n_{sym.lower()}" for sym in self._HETEROATOMS]
        names += [f"frac_{sym.lower()}" for sym in self._HETEROATOMS]
        names += ["n_fragments", "bertz_ct", "labute_asa"]
        return np.array(names)


class SmilesStringFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wyciąga heurystyczne cechy bezpośrednio ze stringa SMILES."""

    def fit(self, X, y=None): return self

    def transform(self, X) -> np.ndarray:
        return np.array([self._featurize(s) for s in X], dtype=np.float32)

    def _featurize(self, smiles: str) -> List[float]:
        if not smiles:
            return [0.0] * 21
        s = smiles
        return [
            float(s.count("@")),
            float(s.count("@@")),
            float(s.count("/") + s.count("\\")),
            float(s.count("+")),
            float(s.count("-")),
            float(abs(s.count("+") - s.count("-"))),
            float(s.count(".")),
            float("[NH" in s or "[nH" in s),
            float("[OH" in s),
            float("[SH" in s),
            float("[PH" in s),
            float("[B" in s),
            float("[Si" in s),
            float("[Se" in s),
            float("[Fe" in s or "[Zn" in s or "[Cu" in s or "[Mg" in s or "[Ca" in s or "[Mn" in s),
            float(len(s)),
            float(s.count("(")),
            float(sum(1 for c in s if c in "cnops")),
            float(sum(1 for c in s if c == "c")),
            float(s.count("=")),
            float(s.count("#")),
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "smiles_n_chiral", "smiles_n_R", "smiles_n_EZ_stereo",
            "smiles_n_pos", "smiles_n_neg", "smiles_charge_imbalance",
            "smiles_n_fragments",
            "smiles_has_NH", "smiles_has_OH", "smiles_has_SH",
            "smiles_has_PH", "smiles_has_B", "smiles_has_Si",
            "smiles_has_Se", "smiles_has_metal",
            "smiles_length", "smiles_n_branches",
            "smiles_n_aromatic_heteroatoms", "smiles_n_aromatic_C",
            "smiles_n_double_bonds", "smiles_n_triple_bonds",
        ])


# ---------------------------------------------------------------------------
# NOWE EKSTRAKTORY v2
# ---------------------------------------------------------------------------

class SaltIonFeatureExtractor(BaseEstimator, TransformerMixin):
    """Cechy soli i jonów wyznaczane na cząsteczce PRZED usunięciem soli.

    Kluczowe dla klas: zwitterion, hydrochloride, ammonium compound,
    organoammonium salt, chloride salt, halide salt, organic salt.

    Przyjmuje mol_original (przed SaltRemover) — stąd musi być podłączony
    przez OriginalMolExtractor.
    """

    def fit(self, X, y=None): return self

    def transform(self, X) -> np.ndarray:
        return np.array([self._featurize(mol) for mol in X], dtype=np.float32)

    def _featurize(self, mol) -> List[float]:
        if mol is None:
            return [0.0] * 20

        frags = Chem.GetMolFrags(mol, asMols=True)
        charges = [Chem.GetFormalCharge(f) for f in frags]
        pos_frags = sum(1 for c in charges if c > 0)
        neg_frags = sum(1 for c in charges if c < 0)
        neutral_frags = sum(1 for c in charges if c == 0)
        total_charge = Chem.GetFormalCharge(mol)

        is_zwitterion = float(total_charge == 0 and pos_frags > 0 and neg_frags > 0)
        is_salt = float(pos_frags > 0 and neg_frags > 0)

        has_chloride_ion = float(
            any("Cl" in Chem.MolToSmiles(f) and Chem.GetFormalCharge(f) < 0
                for f in frags)
        )
        has_Cl = float(any(a.GetSymbol() == "Cl" for a in mol.GetAtoms()))
        has_Br = float(any(a.GetSymbol() == "Br" for a in mol.GetAtoms()))
        has_F  = float(any(a.GetSymbol() == "F"  for a in mol.GetAtoms()))
        has_I  = float(any(a.GetSymbol() == "I"  for a in mol.GetAtoms()))
        is_halide = float(has_Cl or has_Br or has_F or has_I)

        has_ammonium = float(
            any(
                any(a.GetSymbol() == "N" and a.GetFormalCharge() > 0
                    for a in f.GetAtoms())
                for f in frags
            )
        )
        has_organoammonium = float(
            any(
                any(a.GetSymbol() == "N" and a.GetFormalCharge() > 0
                    for a in f.GetAtoms())
                and any(a.GetSymbol() == "C" for a in f.GetAtoms())
                for f in frags
            )
        )

        frag_sizes = sorted([f.GetNumHeavyAtoms() for f in frags], reverse=True)
        largest  = float(frag_sizes[0])  if frag_sizes else 0.0
        smallest = float(frag_sizes[-1]) if frag_sizes else 0.0
        size_ratio = smallest / largest if largest > 0 else 0.0

        return [
            float(len(frags)),
            float(pos_frags),
            float(neg_frags),
            float(neutral_frags),
            float(total_charge),
            float(sum(abs(c) for c in charges)),
            float(max(abs(c) for c in charges) if charges else 0),
            is_salt,
            is_zwitterion,
            has_chloride_ion,
            has_Cl,
            has_Br,
            has_F,
            has_I,
            is_halide,
            has_ammonium,
            has_organoammonium,
            largest,
            smallest,
            size_ratio,
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "ion_n_frags", "ion_pos_frags", "ion_neg_frags", "ion_neutral_frags",
            "ion_total_charge", "ion_sum_abs_charges", "ion_max_abs_charge",
            "ion_is_salt", "ion_is_zwitterion",
            "ion_has_chloride_ion", "ion_has_Cl", "ion_has_Br", "ion_has_F", "ion_has_I",
            "ion_is_halide", "ion_has_ammonium", "ion_has_organoammonium",
            "ion_largest_frag", "ion_smallest_frag", "ion_size_ratio",
        ])


class RingTopologyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Globalna topologia układu pierścieniowego.

    Kluczowe dla klas: organic tetracyclic compound, tetracyclic triterpenoid,
    macrolide, spiro compound, oxaspiro compound, polycyclic compound.
    """

    def fit(self, X, y=None): return self

    def transform(self, X) -> np.ndarray:
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        return np.array([self._featurize(mol) for mol in mols], dtype=np.float32)

    def _featurize(self, mol) -> List[float]:
        if mol is None:
            return [0.0] * 24

        ring_info = mol.GetRingInfo()
        rings = list(ring_info.AtomRings())
        ring_sizes = sorted([len(r) for r in rings], reverse=True)

        largest_ring = max(ring_sizes) if ring_sizes else 0
        n_rings = len(rings)

        has_macrocycle = float(any(s >= 12 for s in ring_sizes))
        n_macrocycles  = float(sum(1 for s in ring_sizes if s >= 12))

        atom_ring_membership = [0] * mol.GetNumAtoms()
        for ring in rings:
            for idx in ring:
                atom_ring_membership[idx] += 1
        n_spiro_atoms = sum(1 for c in atom_ring_membership if c == 2)

        n_fused_pairs = 0
        fused_graph_degree = [0] * n_rings
        for i, r1 in enumerate(rings):
            for j, r2 in enumerate(rings):
                if i < j and len(set(r1) & set(r2)) >= 2:
                    n_fused_pairs += 1
                    fused_graph_degree[i] += 1
                    fused_graph_degree[j] += 1

        n_O_rings = sum(
            1 for ring in rings
            if any(mol.GetAtomWithIdx(a).GetSymbol() == "O" for a in ring)
        )
        n_N_rings = sum(
            1 for ring in rings
            if any(mol.GetAtomWithIdx(a).GetSymbol() == "N" for a in ring)
        )
        n_S_rings = sum(
            1 for ring in rings
            if any(mol.GetAtomWithIdx(a).GetSymbol() == "S" for a in ring)
        )

        n_C_in_rings = len({
            a for ring in rings for a in ring
            if mol.GetAtomWithIdx(a).GetSymbol() == "C"
        })
        total_C = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
        frac_C_in_rings = n_C_in_rings / max(total_C, 1)

        return [
            float(n_rings),
            float(largest_ring),
            has_macrocycle,
            n_macrocycles,
            float(sum(1 for s in ring_sizes if s == 3)),
            float(sum(1 for s in ring_sizes if s == 4)),
            float(sum(1 for s in ring_sizes if s == 5)),
            float(sum(1 for s in ring_sizes if s == 6)),
            float(sum(1 for s in ring_sizes if s == 7)),
            float(sum(1 for s in ring_sizes if s >= 8 and s < 12)),
            float(n_spiro_atoms),
            float(n_spiro_atoms > 0),
            float(n_fused_pairs),
            float(n_fused_pairs >= 3),
            float(n_fused_pairs >= 2),
            float(n_fused_pairs == 1),
            float(max(fused_graph_degree) if fused_graph_degree else 0),
            float(n_O_rings),
            float(n_N_rings),
            float(n_S_rings),
            float(frac_C_in_rings),
            float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
            float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "rt_n_rings", "rt_largest_ring",
            "rt_has_macrocycle", "rt_n_macrocycles",
            "rt_n_ring3", "rt_n_ring4", "rt_n_ring5", "rt_n_ring6",
            "rt_n_ring7", "rt_n_ring8_11",
            "rt_n_spiro_atoms", "rt_is_spiro",
            "rt_n_fused_pairs", "rt_is_tetracyclic", "rt_is_tricyclic",
            "rt_is_bicyclic", "rt_max_fused_degree",
            "rt_n_O_rings", "rt_n_N_rings", "rt_n_S_rings",
            "rt_frac_C_in_rings",
            "rt_spiro_rdkit", "rt_bridgehead_rdkit", "rt_aliphatic_rings",
        ])


class SmartsPatternFeatureExtractor(BaseEstimator, TransformerMixin):
    """Dopasowuje wzorce SMARTS bezpośrednio mapujące definicje klas ChEBI.

    Kluczowe dla klas: pyrazoles, triazoles, benzofurans, aminopyrimidine,
    piperazines, chromenes, anilines, resorcinols, biaryl i inne heterocykle.
    """

    _PATTERNS: Dict[str, str] = {
        # Pięcioczłonowe heterocykle azotowe
        "triazole_1H":     "[nH]1nncc1",
        "triazole_2H":     "n1nn[cH][cH]1",
        "triazole_any":    "n1nncc1",
        "tetrazole":       "c1nnn[nH]1",
        "pyrazole":        "c1cc[nH]n1",
        "pyrazole_sub":    "c1ccnn1",
        "imidazole":       "c1c[nH]cn1",
        "oxazole":         "c1cnco1",
        "isoxazole":       "c1ccno1",
        "thiazole":        "c1cncs1",
        "pyrrole":         "c1cc[nH]c1",
        "indole":          "c1ccc2[nH]ccc2c1",
        "benzimidazole":   "c1ccc2[nH]cnc2c1",
        "purine":          "c1ncnc2[nH]cnc12",
        # Sześcioczłonowe heterocykle azotowe
        "pyrimidine":      "c1cnccn1",
        "aminopyrimidine": "c1cnc(N)nc1",
        "pyridazine":      "c1ccnnc1",
        "pyrazine":        "c1cnccn1",
        "triazine":        "c1ncncn1",
        "quinoline":       "c1ccc2ncccc2c1",
        "isoquinoline":    "c1ccc2cnccc2c1",
        # Heterocykle tlenowe i siarkowe
        "furan":           "c1ccoc1",
        "benzofuran":      "c1coc2ccccc12",
        "benzothiophene":  "c1csc2ccccc12",
        "thiophene":       "c1ccsc1",
        "chromene":        "C1CCc2ccccc2O1",
        "chromene_arom":   "c1ccc2ccoc2c1",
        "coumarin":        "O=C1CCc2ccccc21",
        "morpholine":      "C1CNCCO1",
        "dioxole":         "C1OCCO1",
        # Pierścienie nasycone azotowe
        "piperazine":      "C1CNCCN1",
        "piperidine":      "C1CCNCC1",
        "pyrrolidine":     "C1CCNC1",
        "azetidine":       "C1CCN1",
        # Grupy funkcyjne soli/jonów
        "quaternary_N":    "[N+;!$([N+]-[O-])]",
        "carboxylate":     "C(=O)[O-]",
        "sulfonate":       "S(=O)(=O)[O-]",
        "phosphate":       "P(=O)([O-])",
        "ammonium":        "[NH4+,NH3+,NH2+,NH+]",
        # Grupy kwasów tłuszczowych i alifatyczne
        "long_chain_C6":   "[CH2][CH2][CH2][CH2][CH2][CH2]",
        "long_chain_C8":   "[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2]",
        "carboxylic_acid": "C(=O)[OH]",
        "ester":           "C(=O)OC",
        "lactone":         "[C:1](=O)[O:2][C:3]",
        "lactam":          "[C:1](=O)[N:2][C:3]",
        "thioester":       "C(=O)S",
        # Grupy aromatyczne
        "aniline":         "c1ccccc1N",
        "phenol":          "c1ccccc1O",
        "resorcinol":      "c1cc(O)cc(O)c1",
        "biaryl":          "c1ccccc1-c1ccccc1",
        "naphthalene":     "c1ccc2ccccc2c1",
        "anthracene":      "c1ccc2cc3ccccc3cc2c1",
        # Grupy specjalne
        "epoxide":         "C1OC1",
        "aziridine":       "C1NC1",
        "acetal":          "[CH1]([OR])([OR])",
        "ketal":           "[CR0]([OR])([OR])",
        "allyl":           "C=CC",
        "vinyl":           "C=C",
        "alkyne":          "C#C",
        "nitrile":         "C#N",
        "nitro":           "[N+](=O)[O-]",
        "sulfoxide":       "S(=O)C",
        "sulfone":         "S(=O)(=O)C",
        "guanidine":       "NC(=N)N",
        "urea":            "NC(=O)N",
        "amide":           "C(=O)N",
        "aldehyde":        "[CH]=O",
        "ketone":          "CC(=O)C",
    }

    def __init__(self) -> None:
        self._compiled: Optional[Dict[str, Any]] = None

    def _compile(self) -> None:
        if self._compiled is not None:
            return
        self._compiled = {}
        for name, smarts in self._PATTERNS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None:
                self._compiled[name] = pat
            else:
                log.warning(f"Niepoprawny SMARTS dla '{name}': {smarts}")

    def fit(self, X, y=None):
        self._compile()
        return self

    def transform(self, X) -> np.ndarray:
        self._compile()
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        return np.array([self._featurize(mol) for mol in mols], dtype=np.float32)

    def _featurize(self, mol) -> List[float]:
        if mol is None or self._compiled is None:
            return [0.0] * len(self._PATTERNS)
        return [
            float(mol.HasSubstructMatch(pat))
            for pat in self._compiled.values()
        ]

    def get_feature_names_out(self, input_features=None):
        self._compile()
        return np.array([f"smarts_{name}" for name in self._compiled])


class ChainFeatureExtractor(BaseEstimator, TransformerMixin):
    """Cechy długości i liniowości łańcucha węglowego.

    Kluczowe dla klas: fatty acid, aliphatic compound, alicyclic compound,
    olefin, cyclic olefin, long-chain fatty acid.
    """

    def fit(self, X, y=None): return self

    def transform(self, X) -> np.ndarray:
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        return np.array([self._featurize(mol) for mol in mols], dtype=np.float32)

    def _featurize(self, mol) -> List[float]:
        if mol is None:
            return [0.0] * 14

        try:
            dist_matrix = rdmolops.GetDistanceMatrix(mol)
            max_path = float(dist_matrix.max()) if dist_matrix.size > 0 else 0.0
        except Exception:
            max_path = 0.0

        aliphatic_C = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and not a.GetIsAromatic()
        ]
        n_aliphatic_C = len(aliphatic_C)
        total_heavy = max(mol.GetNumHeavyAtoms(), 1)
        frac_aliphatic_C = n_aliphatic_C / total_heavy

        max_aliphatic_path = 0.0
        if len(aliphatic_C) > 1:
            try:
                for i in aliphatic_C:
                    for j in aliphatic_C:
                        if i < j:
                            max_aliphatic_path = max(
                                max_aliphatic_path, float(dist_matrix[i][j])
                            )
            except Exception:
                pass

        ch2_pat = Chem.MolFromSmarts("[CH2]")
        n_ch2 = len(mol.GetSubstructMatches(ch2_pat)) if ch2_pat else 0

        ch3_pat = Chem.MolFromSmarts("[CH3]")
        n_ch3 = len(mol.GetSubstructMatches(ch3_pat)) if ch3_pat else 0

        cooh_pat = Chem.MolFromSmarts("C(=O)[OH]")
        has_cooh = float(bool(mol.GetSubstructMatches(cooh_pat))) if cooh_pat else 0.0

        fatty_acid_proxy = float(n_ch2 >= 6 and has_cooh)

        n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        is_fully_aliphatic = float(n_arom_rings == 0)

        n_all_rings = rdMolDescriptors.CalcNumRings(mol)
        is_alicyclic = float(n_arom_rings == 0 and n_all_rings > 0)

        olefin_pat = Chem.MolFromSmarts("[CX3]=[CX3]")
        n_olefin = len(mol.GetSubstructMatches(olefin_pat)) if olefin_pat else 0

        return [
            max_path,
            max_aliphatic_path,
            float(n_aliphatic_C),
            frac_aliphatic_C,
            is_fully_aliphatic,
            is_alicyclic,
            float(n_ch2),
            float(n_ch3),
            float(n_ch2 >= 4),
            float(n_ch2 >= 6),
            float(n_ch2 >= 10),
            has_cooh,
            fatty_acid_proxy,
            float(n_olefin),
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "chain_max_path", "chain_max_aliphatic_path",
            "chain_n_aliphatic_C", "chain_frac_aliphatic_C",
            "chain_is_fully_aliphatic", "chain_is_alicyclic",
            "chain_n_ch2", "chain_n_ch3",
            "chain_has_C4plus", "chain_has_C6plus", "chain_has_C10plus",
            "chain_has_cooh", "chain_fatty_acid_proxy",
            "chain_n_olefin",
        ])


class PolymerFeatureExtractor(BaseEstimator, TransformerMixin):
    """Cechy polimeryczne, peptydowe i sacharydowe.

    Kluczowe dla klas: biomacromolecule, trisaccharide derivative,
    oligosaccharide derivative, peptide, glycoside.
    """

    def fit(self, X, y=None): return self

    def transform(self, X) -> np.ndarray:
        mols = [pair[1] if isinstance(pair, tuple) else pair for pair in X]
        return np.array([self._featurize(mol) for mol in mols], dtype=np.float32)

    def _featurize(self, mol) -> List[float]:
        if mol is None:
            return [0.0] * 14

        n_heavy = mol.GetNumHeavyAtoms()

        pyranose_pat   = Chem.MolFromSmarts("C1OC(O)C(O)C(O)C1")
        furanose_pat   = Chem.MolFromSmarts("C1OC(O)C(O)C1")
        glyco_pat      = Chem.MolFromSmarts("C1OCC(O)C(O)C1O")
        peptide_pat    = Chem.MolFromSmarts("[NH]C(=O)")
        glycosidic_pat = Chem.MolFromSmarts("[C;R]O[C;R]")
        phosphate_pat  = Chem.MolFromSmarts("OP(=O)(O)O")

        n_pyranose   = len(mol.GetSubstructMatches(pyranose_pat))   if pyranose_pat   else 0
        n_furanose   = len(mol.GetSubstructMatches(furanose_pat))   if furanose_pat   else 0
        n_glyco      = len(mol.GetSubstructMatches(glyco_pat))      if glyco_pat      else 0
        n_peptide    = len(mol.GetSubstructMatches(peptide_pat))    if peptide_pat    else 0
        n_glycosidic = len(mol.GetSubstructMatches(glycosidic_pat)) if glycosidic_pat else 0
        n_phosphate  = len(mol.GetSubstructMatches(phosphate_pat))  if phosphate_pat  else 0

        return [
            float(n_heavy >= 100),
            float(n_heavy >= 50),
            float(n_heavy >= 30),
            float(n_heavy),
            float(n_pyranose),
            float(n_furanose),
            float(n_glyco),
            float(n_pyranose >= 2),
            float(n_pyranose >= 3),
            float(n_glycosidic),
            float(n_peptide),
            float(n_peptide >= 2),
            float(n_peptide >= 4),
            float(n_phosphate),
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "poly_is_macromol", "poly_is_large", "poly_is_medium", "poly_n_heavy",
            "poly_n_pyranose", "poly_n_furanose", "poly_n_glyco_motif",
            "poly_is_disaccharide", "poly_is_trisaccharide",
            "poly_n_glycosidic", "poly_n_peptide",
            "poly_is_dipeptide", "poly_is_oligopeptide",
            "poly_n_phosphate",
        ])


# ---------------------------------------------------------------------------
# Builder potoku
# ---------------------------------------------------------------------------

def build_chebi_feature_pipeline(config: Optional[FeatureConfig] = None) -> Pipeline:
    """Buduje pełny potok inżynierii cech dla klasyfikacji ChEBI.

    Standaryzator zwraca krotki (mol_original, mol_stripped).
    Każda gałąź FeatureUnion wyciąga potrzebną formę przez dedykowany adapter.

    Args:
        config: Instancja FeatureConfig z flagami. None = wszystko włączone.

    Returns:
        sklearn.pipeline.Pipeline gotowy do fit_transform.
    """
    if config is None:
        config = FeatureConfig()

    transformer_list = []

    # Fingerprints skfp — działają na mol_stripped (przez SkfpMolAdapter)
    if config.use_maccs:
        transformer_list.append(("maccs_keys", SkfpMolAdapter(
            MACCSFingerprint(sparse=True, n_jobs=config.n_jobs)
        )))
        log.info("[FeatureConfig] MACCS Keys: ON")

    if config.use_ecfp_count:
        transformer_list.append(("ecfp_count", SkfpMolAdapter(
            ECFPFingerprint(count=True, radius=config.ecfp_radius, sparse=True, n_jobs=config.n_jobs)
        )))
        log.info("[FeatureConfig] ECFP Count: ON")

    if config.use_atom_pair:
        transformer_list.append(("atom_pair", SkfpMolAdapter(
            AtomPairFingerprint(count=config.atom_pair_count, sparse=True, n_jobs=config.n_jobs)
        )))
        log.info("[FeatureConfig] Atom Pair: ON")

    if config.use_topo_torsion:
        transformer_list.append(("topo_torsion", SkfpMolAdapter(
            TopologicalTorsionFingerprint(count=config.topo_torsion_count, sparse=True, n_jobs=config.n_jobs)
        )))
        log.info("[FeatureConfig] Topological Torsion: ON")

    if config.use_klekota_roth:
        transformer_list.append(("klekota_roth", SkfpMolAdapter(
            KlekotaRothFingerprint(sparse=True, n_jobs=config.n_jobs)
        )))
        log.info("[FeatureConfig] Klekota-Roth: ON")

    if config.use_rdkit_descs:
        transformer_list.append(("rdkit_descriptors", Pipeline([
            ("extract",      SkfpMolAdapter(RDKit2DDescriptorsFingerprint(sparse=False, n_jobs=config.n_jobs))),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] RDKit 2D Descriptors: ON")

    # Cechy domenowe (działają na krotce — wyciągają mol_stripped)
    if config.use_domain_features:
        transformer_list.append(("domain_features", Pipeline([
            ("extract",      DomainFeatureExtractor()),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] Domain Features: ON")

    if config.use_smiles_features:
        transformer_list.append(("smiles_features", Pipeline([
            ("to_smiles",    MolToSmilesTransformer()),
            ("extract",      SmilesStringFeatureExtractor()),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] SMILES String Features: ON")

    # Nowe cechy v2
    if config.use_salt_ion_features:
        transformer_list.append(("salt_ion_features", Pipeline([
            ("orig_mol",     OriginalMolExtractor()),   # mol_original (z solą)
            ("extract",      SaltIonFeatureExtractor()),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] Salt/Ion Features: ON")

    if config.use_ring_topology:
        transformer_list.append(("ring_topology", Pipeline([
            ("extract",      RingTopologyFeatureExtractor()),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] Ring Topology Features: ON")

    if config.use_smarts_patterns:
        transformer_list.append(("smarts_patterns", Pipeline([
            ("extract",      SmartsPatternFeatureExtractor()),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] SMARTS Pattern Features: ON")

    if config.use_chain_features:
        transformer_list.append(("chain_features", Pipeline([
            ("extract",      ChainFeatureExtractor()),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] Chain Features: ON")

    if config.use_polymer_features:
        transformer_list.append(("polymer_features", Pipeline([
            ("extract",      PolymerFeatureExtractor()),
            ("scale_sparse", ScaledDenseToSparse()),
        ])))
        log.info("[FeatureConfig] Polymer Features: ON")

    if not transformer_list:
        raise ValueError(
            "Żadna grupa cech nie jest włączona. "
            "Ustaw co najmniej jedną flagę 'use_*' na True w FeatureConfig."
        )

    feature_union = FeatureUnion(transformer_list=transformer_list, n_jobs=1)

    return Pipeline(steps=[
        ("standardization",      DAGMolStandardizer()),
        ("hybrid_vectorization", feature_union),
    ])


# ---------------------------------------------------------------------------
# Klasa zarządzająca ścieżkami train / valid
# ---------------------------------------------------------------------------

class ChEBIFeaturePipeline:
    """Zarządza osobnymi ścieżkami transformacji dla train i valid.

    Scaler jest dopasowywany wyłącznie na zbiorze treningowym – valid
    jest tylko transformowany, co zapobiega wyciekowi danych.

    Przykład::

        pipeline = ChEBIFeaturePipeline(config)
        X_train = pipeline.fit_transform_train(df_train, smiles_col="SMILES")
        X_valid = pipeline.transform_valid(df_valid, smiles_col="SMILES")
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self._pipeline: Optional[Pipeline] = None

    def fit_transform_train(self, df_train: pd.DataFrame, smiles_col: str = "SMILES") -> csr_matrix:
        log.info("Budowanie i dopasowywanie potoku cech (zbiór treningowy)...")
        log.info(f"Konfiguracja cech: {self.config}")
        self._pipeline = build_chebi_feature_pipeline(self.config)
        X_train = self._pipeline.fit_transform(df_train[smiles_col].tolist())
        if not issparse(X_train):
            X_train = csr_matrix(X_train)
        log.info(f"X_train shape: {X_train.shape}")
        return X_train

    def transform_valid(self, df_valid: pd.DataFrame, smiles_col: str = "SMILES") -> csr_matrix:
        if self._pipeline is None:
            raise RuntimeError("Potok nie jest dopasowany. Wywołaj najpierw fit_transform_train().")
        log.info("Transformacja zbioru walidacyjnego (tylko transform)...")
        X_valid = self._pipeline.transform(df_valid[smiles_col].tolist())
        if not issparse(X_valid):
            X_valid = csr_matrix(X_valid)
        log.info(f"X_valid shape: {X_valid.shape}")
        return X_valid


# ---------------------------------------------------------------------------
# Wygodna funkcja (backward-compatible)
# ---------------------------------------------------------------------------

def preprocess_chebi_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    smiles_col: str = "SMILES",
    config: Optional[FeatureConfig] = None,
) -> Tuple[csr_matrix, csr_matrix]:
    """Wygodna funkcja – przetwarza train i test w jednym wywołaniu.

    Zachowuje kompatybilność wsteczną z poprzednią wersją modułu.

    Args:
        df_train:   Zbiór treningowy z kolumną SMILES.
        df_test:    Zbiór testowy z kolumną SMILES.
        smiles_col: Nazwa kolumny z SMILES.
        config:     FeatureConfig (domyślnie wszystko włączone).

    Returns:
        Tuple (X_train_sparse, X_test_sparse).

    Przykład::

        # Wszystkie cechy włączone (domyślnie):
        X_train, X_test = preprocess_chebi_data(df_train, df_test)

        # Szybki tryb bez Klekota-Roth i nowych v2:
        cfg = FeatureConfig(
            use_klekota_roth=False,
            use_salt_ion_features=False,
            use_ring_topology=False,
            use_smarts_patterns=False,
            use_chain_features=False,
            use_polymer_features=False,
        )
        X_train, X_test = preprocess_chebi_data(df_train, df_test, config=cfg)
    """
    pipeline_obj = ChEBIFeaturePipeline(config)
    X_train = pipeline_obj.fit_transform_train(df_train, smiles_col)
    X_test  = pipeline_obj.transform_valid(df_test, smiles_col)

    if config is not None:
        n_enabled = sum([
            config.use_maccs, config.use_ecfp_count, config.use_atom_pair,
            config.use_topo_torsion, config.use_klekota_roth, config.use_rdkit_descs,
            config.use_domain_features, config.use_smiles_features,
            config.use_salt_ion_features, config.use_ring_topology,
            config.use_smarts_patterns, config.use_chain_features,
            config.use_polymer_features,
        ])
        log.info(f"Aktywne grupy cech: {n_enabled}/13")

    log.info("Preprocessing zakończony sukcesem.")
    return X_train, X_test
