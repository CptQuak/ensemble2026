"""
chebi_training.py
=================
Trening i ewaluacja wieloetykietowego klasyfikatora LightGBM dla ChEBI.

Eksportuje:
    LGBMConfig           – dataclass z hiperparametrami LightGBM
    F1Callback           – callback monitorujący F1 co N rund
    train_multilabel     – trening jednego modelu binarnego na klasę
    train_all_classes    – pętla po wszystkich klasach zwracająca modele + prob matrix
    evaluate             – wyliczenie metryk końcowych
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

@dataclass
class LGBMConfig:
    """Hiperparametry i ustawienia treningu LightGBM.

    Atrybuty:
        n_estimators:  Maksymalna liczba drzew.
        early_stopping_rounds: Rundy bez poprawy → zatrzymanie.
        threshold:     Próg decyzyjny (prob → 0/1).
        f1_log_period: Co ile rund logować F1 w callbacku.
        params:        Słownik parametrów przekazywany do lgb.train.
    """
    n_estimators: int = 300
    early_stopping_rounds: int = 30
    threshold: float = 0.5
    f1_log_period: int = 20

    params: Dict = field(default_factory=lambda: {
        "objective":         "binary",
        "metric":            "binary_logloss",
        "boosting_type":     "gbdt",
        "num_leaves":        63,
        "learning_rate":     0.05,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "min_child_samples": 20,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "is_unbalance":      True,   # obsługa niezbalansowanych klas ChEBI
        "verbose":           -1,
        "n_jobs":            -1,
    })


# ---------------------------------------------------------------------------
# Callback F1
# ---------------------------------------------------------------------------

class F1Callback:
    """Liczy binary F1 na zbiorze walidacyjnym co `period` rund boostingu.

    Atrybuty:
        history: Lista krotek (iteracja, f1_score) zebrana podczas treningu.
    """

    def __init__(
        self,
        X_valid: csr_matrix,
        y_valid_col: np.ndarray,
        period: int = 20,
        threshold: float = 0.5,
    ) -> None:
        self.X_valid     = X_valid
        self.y_valid_col = y_valid_col
        self.period      = period
        self.threshold   = threshold
        self.history: List[Tuple[int, float]] = []

    def __call__(self, env: lgb.callback.CallbackEnv) -> None:
        if (env.iteration + 1) % self.period != 0:
            return
        y_prob = env.model.predict(self.X_valid)
        y_pred = (y_prob >= self.threshold).astype(int)
        f1 = f1_score(self.y_valid_col, y_pred, average="binary", zero_division=0)
        self.history.append((env.iteration + 1, f1))


# ---------------------------------------------------------------------------
# Trening pojedynczej klasy
# ---------------------------------------------------------------------------

def train_single_class(
    X_train: csr_matrix,
    y_train_col: np.ndarray,
    X_valid: csr_matrix,
    y_valid_col: np.ndarray,
    config: LGBMConfig,
) -> Tuple[lgb.Booster, F1Callback]:
    """Trenuje jeden binarny klasyfikator LightGBM dla pojedynczej klasy ChEBI.

    Args:
        X_train:     Macierz cech zbioru treningowego.
        y_train_col: Wektor etykiet binarnych dla klasy (train).
        X_valid:     Macierz cech zbioru walidacyjnego.
        y_valid_col: Wektor etykiet binarnych dla klasy (valid).
        config:      Konfiguracja LightGBM.

    Returns:
        Tuple (wytrenowany model Booster, instancja F1Callback z historią).
    """
    dtrain = lgb.Dataset(X_train, label=y_train_col, free_raw_data=False)
    dvalid  = lgb.Dataset(X_valid,  label=y_valid_col, reference=dtrain, free_raw_data=False)

    f1_cb = F1Callback(X_valid, y_valid_col, period=config.f1_log_period, threshold=config.threshold)

    model = lgb.train(
        params=config.params,
        train_set=dtrain,
        num_boost_round=config.n_estimators,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            f1_cb,
        ],
    )
    return model, f1_cb


# ---------------------------------------------------------------------------
# Pętla treningowa po wszystkich klasach
# ---------------------------------------------------------------------------

def train_all_classes(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_valid: csr_matrix,
    y_valid: np.ndarray,
    label_cols: List[str],
    config: Optional[LGBMConfig] = None,
) -> Tuple[List[lgb.Booster], np.ndarray, Dict[str, float]]:
    """Trenuje osobny klasyfikator binarny dla każdej klasy ChEBI.

    Args:
        X_train:    Macierz cech treningowych (n_train, n_features).
        y_train:    Macierz etykiet treningowych (n_train, n_classes).
        X_valid:    Macierz cech walidacyjnych (n_valid, n_features).
        y_valid:    Macierz etykiet walidacyjnych (n_valid, n_classes).
        label_cols: Lista nazw klas odpowiadających kolumnom y_*.
        config:     Konfiguracja LightGBM. Jeśli None – użyj domyślnej.

    Returns:
        Tuple:
            models           – lista wytrenowanych Boosterów (jeden na klasę),
            prob_matrix_valid – macierz prawdopodobieństw (n_valid, n_classes),
            f1_per_class      – słownik {class_name: f1_score}.
    """
    if config is None:
        config = LGBMConfig()

    n_classes = len(label_cols)
    models: List[lgb.Booster] = []
    f1_per_class: Dict[str, float] = {}
    prob_matrix_valid = np.zeros((X_valid.shape[0], n_classes), dtype=np.float32)

    t0 = time.time()
    pbar = tqdm(range(n_classes), desc="Trening klas", unit="klasa", leave=True)

    for i in pbar:
        class_name = label_cols[i]
        y_tr_col   = y_train[:, i]
        y_va_col   = y_valid[:, i]

        model, _ = train_single_class(X_train, y_tr_col, X_valid, y_va_col, config)
        models.append(model)

        y_prob = model.predict(X_valid).astype(np.float32)
        prob_matrix_valid[:, i] = y_prob

        f1 = f1_score(y_va_col, (y_prob >= config.threshold).astype(int),
                      average="binary", zero_division=0)
        f1_per_class[class_name] = f1

        running_macro = float(np.mean(list(f1_per_class.values())))
        pbar.set_postfix({
            "klasa":    class_name,
            "F1_klasy": f"{f1:.4f}",
            "F1_macro": f"{running_macro:.4f}",
            "drzewa":   model.num_trees(),
        })

    pbar.close()
    elapsed = time.time() - t0
    final_macro = float(np.mean(list(f1_per_class.values())))

    log.info(f"Trening zakończony w {elapsed / 60:.1f} min.")
    log.info(f"Macro-averaged F1 (valid): {final_macro:.4f}")

    return models, prob_matrix_valid, f1_per_class


# ---------------------------------------------------------------------------
# Ewaluacja końcowa
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    prob_matrix: np.ndarray,
    label_cols: List[str],
    threshold: float = 0.5,
    top_n: int = 5,
) -> Dict[str, float]:
    """Wyświetla metryki końcowe i zwraca słownik wyników.

    Args:
        y_true:      Macierz prawdziwych etykiet (n_samples, n_classes).
        prob_matrix: Macierz prawdopodobieństw (n_samples, n_classes).
        label_cols:  Nazwy klas odpowiadające kolumnom.
        threshold:   Próg decyzyjny (domyślnie 0.5).
        top_n:       Ile najlepszych/najgorszych klas wypisać.

    Returns:
        Słownik {'f1_macro': float, 'f1_per_class': Dict[str, float]}.
    """
    y_pred = (prob_matrix >= threshold).astype(np.int8)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    f1_per_class = {
        col: f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
        for i, col in enumerate(label_cols)
    }

    print(f"\n{'=' * 55}")
    print(f"  Macro-averaged F1: {f1_macro:.4f}")
    print(f"{'=' * 55}")

    sorted_f1 = sorted(f1_per_class.items(), key=lambda x: x[1])
    print(f"\nNajgorsze {top_n} klas:")
    for name, score in sorted_f1[:top_n]:
        print(f"  {name}: {score:.4f}")
    print(f"\nNajlepsze {top_n} klas:")
    for name, score in sorted_f1[-top_n:]:
        print(f"  {name}: {score:.4f}")

    return {"f1_macro": f1_macro, "f1_per_class": f1_per_class}
