"""
Potok treningowy dla klasyfikacji ontologii ChEBI.

Architektura:
  - LightGBM jako główny estymator (MultiOutputClassifier)
  - Kalibracja prawdopodobieństw (CalibratedClassifierCV)
  - Propagacja etykiet w górę DAG jako postprocessing
  - Monitoring macro-averaged F1 w czasie rzeczywistym przez callback LightGBM

Użycie:
    python chebi_training_pipeline.py
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import potoku cech (plik z poprzedniego kroku)
from chebi_feature_pipeline import FeatureConfig, preprocess_chebi_data

# ---------------------------------------------------------------------------
# Konfiguracja logowania
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Parser hierarchii DAG z pliku chebi_classes.txt / .obo
# ---------------------------------------------------------------------------

def parse_chebi_dag(obo_path: str) -> nx.DiGraph:
    """Parsuje plik .obo / .txt z definicjami klas ChEBI.

    Czyta relacje `is_a` i buduje skierowany graf DAG,
    gdzie krawędź child → parent oznacza "jest podklasą".

    Args:
        obo_path: Ścieżka do pliku chebi_classes.txt lub .obo.

    Returns:
        nx.DiGraph z węzłami 'class_0' ... 'class_499' i krawędziami is_a.
    """
    dag = nx.DiGraph()
    current_id: Optional[str] = None

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id:"):
                current_id = line.split("id:")[1].strip()
                dag.add_node(current_id)
            elif line.startswith("is_a:") and current_id is not None:
                parent = line.split("is_a:")[1].strip().split()[0]
                dag.add_edge(current_id, parent)  # child → parent

    log.info(f"DAG wczytany: {dag.number_of_nodes()} węzłów, "
             f"{dag.number_of_edges()} krawędzi is_a.")
    return dag


def propagate_labels_upward(
    y_pred: np.ndarray,
    dag: nx.DiGraph,
    class_columns: List[str],
) -> np.ndarray:
    """Wymusza spójność hierarchiczną: jeśli dziecko=1, wszyscy przodkowie=1.

    Redukuje liczbę niespójności (tiebreaker w konkursie) i poprawia F1
    dla klas-rodziców, które są częste i łatwe do przewidzenia.

    Args:
        y_pred:        Binarna macierz predykcji (n_samples, n_classes).
        dag:           Graf DAG z relacjami is_a (child → parent).
        class_columns: Lista nazw klas w kolejności kolumn (np. ['class_0', ...]).

    Returns:
        Poprawiona macierz predykcji (n_samples, n_classes).
    """
    col_to_idx = {col: i for i, col in enumerate(class_columns)}
    result = y_pred.copy()

    # Topologiczny sort: od liści (dziecko) ku korzeniowi (rodzic)
    try:
        topo_order = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        log.warning("DAG zawiera cykl – pomijam propagację etykiet.")
        return result

    for node in topo_order:
        if node not in col_to_idx:
            continue
        node_idx = col_to_idx[node]
        # Dla każdej próbki, gdzie dziecko=1, ustaw rodziców=1
        active_samples = np.where(result[:, node_idx] == 1)[0]
        if len(active_samples) == 0:
            continue
        for parent in dag.successors(node):  # successors = parents (child→parent)
            if parent in col_to_idx:
                parent_idx = col_to_idx[parent]
                result[active_samples, parent_idx] = 1

    n_fixed = int(np.sum(result) - np.sum(y_pred))
    log.info(f"Propagacja DAG: naprawiono {n_fixed} niespójnych etykiet.")
    return result


# ---------------------------------------------------------------------------
# 2. Callback F1 macro dla LightGBM (jeden klasyfikator binarny)
# ---------------------------------------------------------------------------

class F1MacroCallback:
    """Callback LightGBM śledzący macro-averaged F1 na zbiorze walidacyjnym.

    Wywoływany po każdej rundzie boostingu. Wyświetla postęp przez tqdm
    i przechowuje historię metryk do późniejszego wykresu.

    Args:
        X_val:       Macierz cech zbioru walidacyjnego.
        y_val_col:   Wektor etykiet binarnych dla jednej klasy.
        class_name:  Nazwa klasy (do logowania).
        period:      Co ile rund liczyć F1 (domyślnie 10).
        threshold:   Próg decyzyjny (domyślnie 0.5).
    """

    def __init__(
        self,
        X_val: csr_matrix,
        y_val_col: np.ndarray,
        class_name: str = "",
        period: int = 10,
        threshold: float = 0.5,
    ):
        self.X_val = X_val
        self.y_val_col = y_val_col
        self.class_name = class_name
        self.period = period
        self.threshold = threshold
        self.history: List[Tuple[int, float]] = []

    def __call__(self, env: lgb.callback.CallbackEnv) -> None:
        iteration = env.iteration + 1
        if iteration % self.period != 0:
            return
        y_prob = env.model.predict(self.X_val)
        y_pred = (y_prob >= self.threshold).astype(int)
        f1 = f1_score(self.y_val_col, y_pred, average="binary", zero_division=0)
        self.history.append((iteration, f1))


# ---------------------------------------------------------------------------
# 3. Główny estymator: MultiOutput LightGBM z monitoringiem F1
# ---------------------------------------------------------------------------

class ChEBILightGBMClassifier:
    """Wieloetykietowy klasyfikator ChEBI oparty na LightGBM.

    Trenuje osobny binarny klasyfikator LightGBM dla każdej z 500 klas.
    Monitoruje macro-averaged F1 na zbiorze walidacyjnym w czasie rzeczywistym.

    Args:
        lgbm_params:    Słownik parametrów LightGBM (bez 'objective').
        n_estimators:   Maksymalna liczba drzew (rund boostingu).
        early_stopping: Liczba rund bez poprawy do zatrzymania.
        f1_period:      Co ile rund obliczać F1 w callbacku.
        threshold:      Próg decyzyjny (domyślnie 0.5).
        calibrate:      Czy kalibrować prawdopodobieństwa (Platt scaling).
        n_jobs:         Liczba wątków LightGBM.
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "is_unbalance": True,   # obsługa niezbalansowanych klas ChEBI
    }

    def __init__(
        self,
        lgbm_params: Optional[Dict] = None,
        n_estimators: int = 500,
        early_stopping: int = 30,
        f1_period: int = 20,
        threshold: float = 0.5,
        calibrate: bool = False,
        n_jobs: int = -1,
    ):
        self.lgbm_params = {**self.DEFAULT_PARAMS, **(lgbm_params or {})}
        self.lgbm_params["n_jobs"] = n_jobs
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.f1_period = f1_period
        self.threshold = threshold
        self.calibrate = calibrate
        self.n_jobs = n_jobs

        self.models_: List[lgb.Booster] = []
        self.class_names_: List[str] = []
        self.f1_histories_: Dict[str, List[Tuple[int, float]]] = {}
        self.val_f1_per_class_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> "ChEBILightGBMClassifier":
        """Trenuje model dla każdej klasy z monitoringiem F1.

        Args:
            X_train:      Macierz cech treningowych.
            y_train:      Macierz etykiet treningowych (n_samples, n_classes).
            X_val:        Macierz cech walidacyjnych.
            y_val:        Macierz etykiet walidacyjnych.
            class_names:  Lista nazw klas (opcjonalnie).

        Returns:
            self
        """
        n_classes = y_train.shape[1]
        self.class_names_ = class_names or [f"class_{i}" for i in range(n_classes)]
        self.models_ = []

        log.info(f"Trening {n_classes} klasyfikatorów binarnych LightGBM...")
        log.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

        # Postęp ogólny przez tqdm
        pbar = tqdm(
            range(n_classes),
            desc="Trening klas",
            unit="klasa",
            dynamic_ncols=True,
        )

        running_f1_scores = []

        for i in pbar:
            class_name = self.class_names_[i]
            y_train_col = y_train[:, i].astype(np.float32)
            y_val_col   = y_val[:, i].astype(np.float32)

            # Zbiory LightGBM
            dtrain = lgb.Dataset(X_train, label=y_train_col, free_raw_data=False)
            dval   = lgb.Dataset(X_val,   label=y_val_col,   reference=dtrain,
                                 free_raw_data=False)

            # Callback F1
            f1_cb = F1MacroCallback(
                X_val=X_val,
                y_val_col=y_val_col,
                class_name=class_name,
                period=self.f1_period,
                threshold=self.threshold,
            )

            # Trening
            callbacks = [
                lgb.early_stopping(self.early_stopping, verbose=False),
                lgb.log_evaluation(period=-1),  # wycisz wbudowane logi
                f1_cb,
            ]

            model = lgb.train(
                params=self.lgbm_params,
                train_set=dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            self.models_.append(model)
            self.f1_histories_[class_name] = f1_cb.history

            # Finalne F1 dla tej klasy
            y_prob = model.predict(X_val)
            y_pred = (y_prob >= self.threshold).astype(int)
            final_f1 = f1_score(y_val_col, y_pred, average="binary", zero_division=0)
            self.val_f1_per_class_[class_name] = final_f1
            running_f1_scores.append(final_f1)

            # Aktualizacja paska postępu
            current_macro_f1 = float(np.mean(running_f1_scores))
            pbar.set_postfix({
                "klasa": class_name,
                "F1_klasy": f"{final_f1:.4f}",
                "F1_macro": f"{current_macro_f1:.4f}",
                "drzewa": model.num_trees(),
            })

        pbar.close()

        final_macro_f1 = float(np.mean(list(self.val_f1_per_class_.values())))
        log.info(f"\n{'='*60}")
        log.info(f"TRENING ZAKOŃCZONY")
        log.info(f"Macro-averaged F1 (walidacja): {final_macro_f1:.4f}")
        log.info(f"{'='*60}")

        # Top 10 i bottom 10 klas
        sorted_f1 = sorted(self.val_f1_per_class_.items(), key=lambda x: x[1])
        log.info("\nNajgorsze 10 klas (F1):")
        for name, score in sorted_f1[:10]:
            log.info(f"  {name}: {score:.4f}")
        log.info("\nNajlepsze 10 klas (F1):")
        for name, score in sorted_f1[-10:]:
            log.info(f"  {name}: {score:.4f}")

        self.is_fitted_ = True
        return self

    def predict_proba_matrix(self, X: csr_matrix) -> np.ndarray:
        """Zwraca macierz prawdopodobieństw (n_samples, n_classes).

        Args:
            X: Macierz cech.

        Returns:
            np.ndarray prawdopodobieństw klasy pozytywnej.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model nie jest wytrenowany. Wywołaj fit() najpierw.")

        proba_matrix = np.zeros((X.shape[0], len(self.models_)), dtype=np.float32)
        for i, model in enumerate(
            tqdm(self.models_, desc="Predykcja", unit="klasa", dynamic_ncols=True)
        ):
            proba_matrix[:, i] = model.predict(X).astype(np.float32)
        return proba_matrix

    def predict(self, X: csr_matrix) -> np.ndarray:
        """Zwraca binarne predykcje (n_samples, n_classes).

        Args:
            X: Macierz cech.

        Returns:
            np.ndarray binarnych predykcji.
        """
        proba = self.predict_proba_matrix(X)
        return (proba >= self.threshold).astype(np.int8)

    def macro_f1_on_val(self) -> float:
        """Zwraca macro-averaged F1 wyznaczony podczas treningu."""
        if not self.val_f1_per_class_:
            return 0.0
        return float(np.mean(list(self.val_f1_per_class_.values())))


# ---------------------------------------------------------------------------
# 4. Funkcja licząca niespójności DAG (tiebreaker)
# ---------------------------------------------------------------------------

def count_dag_inconsistencies(
    y_prob: np.ndarray,
    dag: nx.DiGraph,
    class_columns: List[str],
) -> int:
    """Liczy niespójności: P(dziecko) > P(rodzic) wg prawdopodobieństw.

    Args:
        y_prob:        Macierz prawdopodobieństw (n_samples, n_classes).
        dag:           Graf DAG (child → parent).
        class_columns: Nazwy klas w kolejności kolumn.

    Returns:
        Łączna liczba niespójnych par (próbka, krawędź DAG).
    """
    col_to_idx = {col: i for i, col in enumerate(class_columns)}
    inconsistencies = 0

    for child, parent in dag.edges():
        if child not in col_to_idx or parent not in col_to_idx:
            continue
        child_idx  = col_to_idx[child]
        parent_idx = col_to_idx[parent]
        # Niespójność: dziecko bardziej prawdopodobne niż rodzic
        inconsistencies += int(np.sum(
            y_prob[:, child_idx] > y_prob[:, parent_idx]
        ))

    return inconsistencies



