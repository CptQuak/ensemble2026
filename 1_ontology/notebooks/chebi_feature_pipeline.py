"""
Moduł inżynierii cech dla klasyfikacji ontologii ChEBI.

Zawiera rozszerzony potok cech z możliwością włączania/wyłączania
poszczególnych grup cech poprzez flagi konfiguracyjne.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.base import BaseEstimator, TransformerMixin
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


# ---------------------------------------------------------------------------
# Konfiguracja flag cech
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Konfiguracja włączania/wyłączania poszczególnych grup cech.

    Atrybuty:
        use_maccs:           MACCS Keys – 166 predefinowanych grup funkcyjnych.
        use_ecfp_count:      ECFP Count – lokalne otoczenia promieniowe (hashed).
        use_atom_pair:       Atom Pair – odległości topologiczne między parami atomów.
        use_topo_torsion:    Topological Torsion – czwórki atomów, kąty skręcenia.
        use_klekota_roth:    Klekota-Roth – 4860 wzorców farmakoforycznych.
        use_rdkit_descs:     RDKit 2D Descriptors – ~200 właściwości fizykochemicznych.
        use_domain_features: Ręcznie wyznaczone cechy domenowe (pierścienie, ładunek itp.).
        use_smiles_features: Cechy ze stringa SMILES (sole, chiralność, jony itp.).

        ecfp_radius:         Promień ECFP (domyślnie 2 → ECFP4).
        ecfp_n_bits:         Liczba bitów ECFP (gdy count=False), ignorowane przy count=True.
        atom_pair_count:     Czy używać wersji zliczeniowej Atom Pair.
        topo_torsion_count:  Czy używać wersji zliczeniowej Topological Torsion.
        n_jobs:              Liczba wątków dla fingerprints scikit-fingerprints.
    """
    # --- Flagi grup cech ---
    use_maccs: bool = True
    use_ecfp_count: bool = True
    use_atom_pair: bool = True
    use_topo_torsion: bool = True
    use_klekota_roth: bool = True
    use_rdkit_descs: bool = True
    use_domain_features: bool = True
    use_smiles_features: bool = True

    # --- Hiperparametry fingerprints ---
    ecfp_radius: int = 2
    ecfp_n_bits: int = 2048
    atom_pair_count: bool = True
    topo_torsion_count: bool = True

    # --- Parametry obliczeniowe ---
    n_jobs: int = -1


# ---------------------------------------------------------------------------
# Standardyzator molekularny
# ---------------------------------------------------------------------------

class DAGMolStandardizer(BaseEstimator, TransformerMixin):
    """Niskopoziomowy transformator RDKit przygotowujący grafy pod klasyfikację ChEBI.

    Parsuje SMILES, weryfikuje poprawność, usuwa sole. Nie filtruje dużych
    molekuł – ontologia ChEBI zawiera makromolekuły i złożone peptydy.

    Atrybuty:
        salt_remover: Instancja SaltRemover do odcinania asocjatów jonowych.
    """

    def __init__(self) -> None:
        self.salt_remover = SaltRemover()

    def fit(self, X: List[str], y: Optional[Any] = None) -> 'DAGMolStandardizer':
        return self

    def transform(self, X: List[str]) -> List[Optional[Chem.Mol]]:
        """Aplikuje standaryzację na grafach molekularnych.

        Args:
            X: Lista ciągów SMILES.

        Returns:
            Lista wyczyszczonych obiektów rdkit.Chem.Mol (None dla błędnych SMILES).
        """
        standardized_mols = []
        for smiles in X:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logging.warning(f"Odrzucono uszkodzony SMILES: {smiles}")
                standardized_mols.append(None)
                continue
            clean_mol = self.salt_remover.StripMol(mol, dontRemoveEverything=True)
            if clean_mol is not None:
                try:
                    Chem.SanitizeMol(clean_mol)
                except Exception as e:
                    logging.warning(f"Błąd sanityzacji: {e}. Używam oryginału.")
                    clean_mol = mol
            standardized_mols.append(clean_mol)
        return standardized_mols


# ---------------------------------------------------------------------------
# Transformatory cech domenowych
# ---------------------------------------------------------------------------

class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wyznacza ręcznie zaprojektowane cechy domenowe z grafów molekularnych.

    Cechy obejmują skład atomowy, topologię pierścieni, właściwości
    fizykochemiczne oraz cechy stereochemiczne – kluczowe dla hierarchii ChEBI.

    Wszystkie cechy są ciągłe i wymagają późniejszej normalizacji (StandardScaler).
    """

    # Symbole atomów, dla których zliczamy wystąpienia
    _HETEROATOMS = ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si', 'Se', 'B']

    def fit(self, X: List[Optional[Chem.Mol]], y: Optional[Any] = None) -> 'DomainFeatureExtractor':
        return self

    def transform(self, X: List[Optional[Chem.Mol]]) -> np.ndarray:
        """Wyznacza macierz cech domenowych.

        Args:
            X: Lista obiektów rdkit.Chem.Mol (None zastępowane zerami).

        Returns:
            np.ndarray o kształcie (n_samples, n_domain_features).
        """
        return np.array([self._featurize(mol) for mol in X], dtype=np.float32)

    def _featurize(self, mol: Optional[Chem.Mol]) -> List[float]:
        if mol is None:
            return [0.0] * self._n_features()

        feats = []

        # --- Ładunek formalny (zwitteriony, jony, chelaty) ---
        feats.append(float(Chem.GetFormalCharge(mol)))
        feats.append(float(abs(Chem.GetFormalCharge(mol))))
        feats.append(float(sum(
            abs(a.GetFormalCharge()) for a in mol.GetAtoms()
        )))  # sumaryczny ładunek bezwzględny

        # --- Rodniki (np. klasa "radical") ---
        feats.append(float(Descriptors.NumRadicalElectrons(mol)))

        # --- Topologia pierścieni ---
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

        # Rozkład wielkości pierścieni (3-8+ członowy)
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        for size in range(3, 9):
            feats.append(float(sum(1 for s in ring_sizes if s == size)))
        feats.append(float(sum(1 for s in ring_sizes if s >= 9)))  # makrocykle

        # --- Stopień nasycenia i hybrydyzacja ---
        feats.append(float(rdMolDescriptors.CalcFractionCSP3(mol)))
        n_atoms = mol.GetNumAtoms()
        if n_atoms > 0:
            sp3 = sum(1 for a in mol.GetAtoms()
                      if a.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
            sp2 = sum(1 for a in mol.GetAtoms()
                      if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2)
            sp  = sum(1 for a in mol.GetAtoms()
                      if a.GetHybridization() == Chem.rdchem.HybridizationType.SP)
            feats.extend([sp3 / n_atoms, sp2 / n_atoms, sp / n_atoms])
        else:
            feats.extend([0.0, 0.0, 0.0])

        # --- Stereochemia ---
        feats.append(float(rdMolDescriptors.CalcNumAtomStereoCenters(mol)))
        feats.append(float(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)))
        stereo_bonds = sum(
            1 for b in mol.GetBonds()
            if b.GetStereo() not in (
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY,
            )
        )
        feats.append(float(stereo_bonds))

        # --- Właściwości fizykochemiczne (proxy pKa / ADME) ---
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

        # --- Wiązania ---
        bonds = mol.GetBonds()
        n_single = sum(1 for b in bonds if b.GetBondTypeAsDouble() == 1.0)
        n_double = sum(1 for b in bonds if b.GetBondTypeAsDouble() == 2.0)
        n_triple = sum(1 for b in bonds if b.GetBondTypeAsDouble() == 3.0)
        n_arom   = sum(1 for b in bonds if b.GetIsAromatic())
        n_bonds  = mol.GetNumBonds()
        feats.extend([float(n_single), float(n_double),
                      float(n_triple), float(n_arom)])
        if n_bonds > 0:
            feats.append(float(n_arom) / n_bonds)  # frakcja wiązań aromatycznych
        else:
            feats.append(0.0)

        # --- Skład atomowy (heteroatomy) ---
        atom_counts = {sym: 0 for sym in self._HETEROATOMS}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in atom_counts:
                atom_counts[sym] += 1
        feats.extend([float(atom_counts[sym]) for sym in self._HETEROATOMS])

        # Frakcje heteroatomów względem liczby atomów ciężkich
        n_heavy = max(mol.GetNumHeavyAtoms(), 1)
        for sym in self._HETEROATOMS:
            feats.append(atom_counts[sym] / n_heavy)

        # --- Informacja o fragmentach (sole, kompleksy) ---
        frags = Chem.GetMolFrags(mol)
        feats.append(float(len(frags)))  # liczba niezależnych fragmentów

        # --- Złożoność molekularna ---
        feats.append(float(Descriptors.BertzCT(mol)))  # indeks złożoności Bertza
        feats.append(float(rdMolDescriptors.CalcLabuteASA(mol)))  # ASA Labute'a

        return feats

    def _n_features(self) -> int:
        """Zwraca oczekiwaną liczbę cech (potrzebne do obsługi None)."""
        # Uruchom raz na prostej molekule żeby policzyć
        dummy = Chem.MolFromSmiles('C')
        return len(self._featurize(dummy))

    def get_feature_names_out(self, input_features=None):
        names = [
            'formal_charge', 'abs_formal_charge', 'sum_abs_atom_charges',
            'n_radical_electrons',
            'n_rings', 'n_aromatic_rings', 'n_saturated_rings',
            'n_aliphatic_rings', 'n_heterocycles',
            'n_aromatic_heterocycles', 'n_saturated_heterocycles',
            'n_spiro_atoms', 'n_bridgehead_atoms',
            'n_ring_size_3', 'n_ring_size_4', 'n_ring_size_5',
            'n_ring_size_6', 'n_ring_size_7', 'n_ring_size_8', 'n_ring_size_9plus',
            'frac_csp3', 'frac_sp3', 'frac_sp2', 'frac_sp',
            'n_stereocenters', 'n_unspec_stereocenters', 'n_stereo_bonds',
            'mol_wt', 'exact_mol_wt', 'tpsa', 'logp',
            'n_hdonors', 'n_hacceptors', 'n_rotatable_bonds',
            'ring_count', 'heavy_atom_count', 'n_valence_electrons',
            'max_partial_charge', 'min_partial_charge',
            'n_single_bonds', 'n_double_bonds', 'n_triple_bonds',
            'n_aromatic_bonds', 'frac_aromatic_bonds',
        ]
        names += [f'n_{sym.lower()}' for sym in self._HETEROATOMS]
        names += [f'frac_{sym.lower()}' for sym in self._HETEROATOMS]
        names += ['n_fragments', 'bertz_ct', 'labute_asa']
        return np.array(names)


class SmilesStringFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wyciąga cechy bezpośrednio ze stringa SMILES bez parsowania grafu.

    Szybkie sygnały heurystyczne, np. obecność soli (`.`), jonów (`+`/`-`),
    chiralności (`@`), geometrii wiązań (`/`, `\\`), długości łańcucha.
    """

    def fit(self, X: List[str], y: Optional[Any] = None) -> 'SmilesStringFeatureExtractor':
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        return np.array([self._featurize(s) for s in X], dtype=np.float32)

    def _featurize(self, smiles: str) -> List[float]:
        if not smiles:
            return [0.0] * 20

        s = smiles
        feats = [
            # Stereochemia i geometria
            float(s.count('@')),                        # centra chiralności (łącznie)
            float(s.count('@@')),                       # R vs S proxy
            float(s.count('/') + s.count('\\')),        # stereo wiązań E/Z

            # Jony i ładunek
            float(s.count('+')),                        # atomy naładowane +
            float(s.count('-')),                        # atomy naładowane -
            float(abs(s.count('+') - s.count('-'))),    # nierównowaga ładunku

            # Sole i kompleksy (fragmenty)
            float(s.count('.')),                        # liczba fragmentów - 1

            # Grupy atomowe obecność
            float('[NH' in s or '[nH' in s),            # protonowany azot
            float('[OH' in s),                          # protonowany tlen
            float('[SH' in s),                          # tiol
            float('[PH' in s),                          # fosfin
            float('[B' in s),                           # bor
            float('[Si' in s),                          # krzem
            float('[Se' in s),                          # selen
            float('[Fe' in s or '[Zn' in s or '[Cu' in s
                   or '[Mg' in s or '[Ca' in s or '[Mn' in s),  # metale

            # Długość i złożoność
            float(len(s)),                              # długość stringa (proxy rozmiaru)
            float(s.count('(')),                        # liczba rozgałęzień

            # Aromatyczność (litery małe = aromatyczne w SMILES)
            float(sum(1 for c in s if c in 'cnops')),   # aromatyczne heteroatomy
            float(sum(1 for c in s if c == 'c')),       # węgiel aromatyczny

            # Wiązania wielokrotne
            float(s.count('=')),                        # wiązania podwójne
            float(s.count('#')),                        # wiązania potrójne
        ]
        return feats

    def get_feature_names_out(self, input_features=None):
        return np.array([
            'smiles_n_chiral', 'smiles_n_R', 'smiles_n_EZ_stereo',
            'smiles_n_pos', 'smiles_n_neg', 'smiles_charge_imbalance',
            'smiles_n_fragments',
            'smiles_has_NH', 'smiles_has_OH', 'smiles_has_SH',
            'smiles_has_PH', 'smiles_has_B', 'smiles_has_Si',
            'smiles_has_Se', 'smiles_has_metal',
            'smiles_length', 'smiles_n_branches',
            'smiles_n_aromatic_heteroatoms', 'smiles_n_aromatic_C',
            'smiles_n_double_bonds', 'smiles_n_triple_bonds',
        ])


# ---------------------------------------------------------------------------
# Pomocnicze transformatory adaptera (mol ↔ SMILES)
# ---------------------------------------------------------------------------

class MolToSmilesTransformer(BaseEstimator, TransformerMixin):
    """Konwertuje listę Mol z powrotem na SMILES (potrzebne dla SmilesStringFeatureExtractor)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[Optional[Chem.Mol]]) -> List[str]:
        result = []
        for mol in X:
            if mol is None:
                result.append('')
            else:
                result.append(Chem.MolToSmiles(mol) or '')
        return result


class DenseToSparse(BaseEstimator, TransformerMixin):
    """Konwertuje gęstą macierz numpy na sparse CSR (potrzebne dla FeatureUnion)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> csr_matrix:
        return csr_matrix(X)


class ScaledDenseToSparse(BaseEstimator, TransformerMixin):
    """Normalizuje gęstą macierz i konwertuje do sparse CSR.

    Czyści dane wejściowe z wartości NaN oraz nieskończonych, zastępując je
    średnią (imputacja), co zapobiega błędom float32 w StandardScaler.
    """

    def __init__(self) -> None:
        """Inicjalizuje komponenty czyszczące i skalujące."""
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def _clean_input(self, X: np.ndarray) -> np.ndarray:
        """Zastępuje nieskończoności (inf) wartościami NaN do imputacji.

        Args:
            X: Macierz wejściowa z potencjalnymi wartościami inf/NaN.

        Returns:
            Oczyszczona macierz numpy.
        """
        X = np.array(X, copy=True)
        X[np.isinf(X)] = np.nan
        return X

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> 'ScaledDenseToSparse':
        """Dopasowuje imputer i scaler do danych.

        Args:
            X: Gęsta macierz cech.
            y: Ignorowane.

        Returns:
            Instancja ScaledDenseToSparse.
        """
        X_clean = self._clean_input(X)
        X_imputed = self.imputer.fit_transform(X_clean)
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X: np.ndarray) -> csr_matrix:
        """Transformuje, czyści i konwertuje dane do formatu rzadkiego.

        Args:
            X: Gęsta macierz cech.

        Returns:
            Macierz rzadka CSR gotowa do FeatureUnion.
        """
        X_clean = self._clean_input(X)
        X_imputed = self.imputer.transform(X_clean)
        X_scaled = self.scaler.transform(X_imputed)
        return csr_matrix(X_scaled)


# ---------------------------------------------------------------------------
# Builder potoku
# ---------------------------------------------------------------------------

def build_chebi_feature_pipeline(config: Optional[FeatureConfig] = None) -> Pipeline:
    """Buduje pełny potok inżynierii cech dla klasyfikacji ChEBI.

    Składa fingerprints i cechy domenowe w FeatureUnion. Cechy ciągłe
    (deskryptory RDKit, domain features, smiles features) są normalizowane
    StandardScalerem przed złączeniem ze sparsowymi fingerprints.

    Args:
        config: Instancja FeatureConfig z flagami włączania grup cech.
                Jeśli None, używa wartości domyślnych (wszystko włączone).

    Returns:
        sklearn.pipeline.Pipeline gotowy do fit_transform.
    """
    if config is None:
        config = FeatureConfig()

    transformer_list = []

    # ------------------------------------------------------------------
    # 1. Fingerprints binarno-zliczeniowe (już sparse z skfp)
    # ------------------------------------------------------------------
    if config.use_maccs:
        transformer_list.append((
            'maccs_keys',
            MACCSFingerprint(sparse=True, n_jobs=config.n_jobs)
        ))
        logging.info("[FeatureConfig] MACCS Keys: ON")

    if config.use_ecfp_count:
        transformer_list.append((
            'ecfp_count',
            ECFPFingerprint(
                count=True,
                radius=config.ecfp_radius,
                sparse=True,
                n_jobs=config.n_jobs,
            )
        ))
        logging.info("[FeatureConfig] ECFP Count: ON")

    if config.use_atom_pair:
        transformer_list.append((
            'atom_pair',
            AtomPairFingerprint(
                count=config.atom_pair_count,
                sparse=True,
                n_jobs=config.n_jobs,
            )
        ))
        logging.info("[FeatureConfig] Atom Pair: ON")

    if config.use_topo_torsion:
        transformer_list.append((
            'topo_torsion',
            TopologicalTorsionFingerprint(
                count=config.topo_torsion_count,
                sparse=True,
                n_jobs=config.n_jobs,
            )
        ))
        logging.info("[FeatureConfig] Topological Torsion: ON")

    if config.use_klekota_roth:
        transformer_list.append((
            'klekota_roth',
            KlekotaRothFingerprint(sparse=True, n_jobs=config.n_jobs)
        ))
        logging.info("[FeatureConfig] Klekota-Roth: ON")

    # ------------------------------------------------------------------
    # 2. Cechy ciągłe wymagające StandardScaler → konwersja do sparse
    # ------------------------------------------------------------------
    if config.use_rdkit_descs:
        transformer_list.append((
            'rdkit_descriptors',
            Pipeline([
                ('fp', RDKit2DDescriptorsFingerprint(sparse=False, n_jobs=config.n_jobs)),
                ('scale_sparse', ScaledDenseToSparse()),
            ])
        ))
        logging.info("[FeatureConfig] RDKit 2D Descriptors: ON")

    if config.use_domain_features:
        transformer_list.append((
            'domain_features',
            Pipeline([
                ('extract', DomainFeatureExtractor()),
                ('scale_sparse', ScaledDenseToSparse()),
            ])
        ))
        logging.info("[FeatureConfig] Domain Features: ON")

    if config.use_smiles_features:
        transformer_list.append((
            'smiles_features',
            Pipeline([
                ('mol_to_smiles', MolToSmilesTransformer()),
                ('extract', SmilesStringFeatureExtractor()),
                ('scale_sparse', ScaledDenseToSparse()),
            ])
        ))
        logging.info("[FeatureConfig] SMILES String Features: ON")

    if not transformer_list:
        raise ValueError(
            "Żadna grupa cech nie jest włączona. "
            "Ustaw co najmniej jedną flagę 'use_*' na True w FeatureConfig."
        )

    feature_union = FeatureUnion(
        transformer_list=transformer_list,
        n_jobs=1,  # n_jobs=-1 w FeatureUnion + n_jobs=-1 w fingerprints = oversubscription
    )

    pipeline = Pipeline(steps=[
        ('standardization', DAGMolStandardizer()),
        ('hybrid_vectorization', feature_union),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# Główna funkcja preprocessingu
# ---------------------------------------------------------------------------

def preprocess_chebi_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    smiles_col: str = 'SMILES',
    config: Optional[FeatureConfig] = None,
) -> Tuple[csr_matrix, csr_matrix]:
    """Wykonuje pełny preprocessing cech dla podzielonych zbiorów danych.

    Proces obejmuje:
      - standaryzację RDKit (usunięcie soli, sanityzacja),
      - generowanie fingerprints wybranych przez config (MACCS, ECFP, Atom Pair,
        Topological Torsion, Klekota-Roth),
      - obliczenie deskryptorów RDKit 2D,
      - ręcznie zaprojektowane cechy domenowe (pierścienie, ładunek, skład),
      - heurystyczne cechy ze stringa SMILES,
      - normalizację (StandardScaler) cech ciągłych.

    Scaler jest dopasowywany wyłącznie na zbiorze treningowym – zbiór testowy
    jest tylko transformowany, co zapobiega wyciekowi danych.

    Args:
        df_train: Zbiór treningowy z kolumną SMILES i etykietami.
        df_test:  Zbiór testowy z kolumną SMILES.
        smiles_col: Nazwa kolumny zawierającej SMILES.
        config: Instancja FeatureConfig z flagami grup cech.
                Jeśli None, włączone są wszystkie grupy.

    Returns:
        Tuple (X_train_sparse, X_test_sparse) – macierze csr_matrix.

    Przykład użycia::

        # Wszystkie cechy włączone (domyślnie):
        X_train, X_test = preprocess_chebi_data(df_train, df_test)

        # Tylko MACCS + ECFP + cechy domenowe:
        cfg = FeatureConfig(
            use_atom_pair=False,
            use_topo_torsion=False,
            use_klekota_roth=False,
            use_rdkit_descs=False,
            use_smiles_features=False,
        )
        X_train, X_test = preprocess_chebi_data(df_train, df_test, config=cfg)
    """
    if config is None:
        config = FeatureConfig()

    logging.info("Inicjalizacja preprocess_pipeline...")
    logging.info(f"Konfiguracja cech: {config}")

    pipeline = build_chebi_feature_pipeline(config)

    # Dopasowanie i transformacja zbioru treningowego
    logging.info("Przetwarzanie zbioru treningowego (X_train)...")
    X_train_sparse = pipeline.fit_transform(df_train[smiles_col].tolist())

    # Tylko transformacja zbioru testowego
    logging.info("Przetwarzanie zbioru testowego (X_test)...")
    X_test_sparse = pipeline.transform(df_test[smiles_col].tolist())

    # Upewnij się, że wyniki są w formacie csr_matrix
    if not issparse(X_train_sparse):
        X_train_sparse = csr_matrix(X_train_sparse)
    if not issparse(X_test_sparse):
        X_test_sparse = csr_matrix(X_test_sparse)

    logging.info("Preprocessing zakończony sukcesem.")
    logging.info(f"X_train shape: {X_train_sparse.shape}")
    logging.info(f"X_test  shape: {X_test_sparse.shape}")

    n_enabled = sum([
        config.use_maccs, config.use_ecfp_count, config.use_atom_pair,
        config.use_topo_torsion, config.use_klekota_roth,
        config.use_rdkit_descs, config.use_domain_features, config.use_smiles_features,
    ])
    logging.info(f"Aktywne grupy cech: {n_enabled}/8")

    return X_train_sparse, X_test_sparse
