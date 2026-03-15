"""
chebi_dag.py
============
Obsługa DAG ontologii ChEBI: parsowanie pliku OBO, propagacja etykiet
w górę hierarchii oraz liczenie niespójności predykcji.

Eksportuje:
    ChEBIDAG  – klasa opakowująca graf nx.DiGraph z metodami pomocniczymi
"""

from __future__ import annotations

import logging
from typing import List

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)


class ChEBIDAG:
    """Opakowuje DAG ontologii ChEBI z metodami do postprocessingu predykcji.

    Atrybuty:
        graph: Skierowany graf (nx.DiGraph) z krawędziami child → parent.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    # ------------------------------------------------------------------
    # Fabryka
    # ------------------------------------------------------------------

    @classmethod
    def from_obo(cls, obo_path: str) -> "ChEBIDAG":
        """Buduje DAG z relacji is_a w pliku OBO.

        Krawędź: child → parent (kierunek propagacji etykiet w górę).

        Args:
            obo_path: Ścieżka do pliku chebi_classes.txt / .obo.

        Returns:
            Instancja ChEBIDAG.
        """
        dag = nx.DiGraph()
        current_id = None

        with open(obo_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    current_id = None
                elif line.startswith("id:"):
                    current_id = line.split("id:", 1)[1].strip()
                    dag.add_node(current_id)
                elif line.startswith("is_a:") and current_id:
                    parent = line.split("is_a:", 1)[1].strip().split()[0]
                    dag.add_edge(current_id, parent)

        log.info(
            f"DAG wczytany z '{obo_path}': "
            f"{dag.number_of_nodes()} węzłów, {dag.number_of_edges()} krawędzi."
        )
        return cls(dag)

    # ------------------------------------------------------------------
    # Propagacja etykiet
    # ------------------------------------------------------------------

    def propagate_labels_upward(
        self,
        y_pred: np.ndarray,
        class_columns: List[str],
    ) -> np.ndarray:
        """Jeśli dziecko=1, wszyscy jego przodkowie w DAG są ustawiani na 1.

        Redukuje niespójności hierarchiczne w predykcjach multi-label.

        Args:
            y_pred:        Macierz predykcji binarnych (n_samples, n_classes).
            class_columns: Lista nazw klas odpowiadających kolumnom y_pred.

        Returns:
            Poprawiona macierz predykcji (kopia oryginału).
        """
        col_to_idx = {col: i for i, col in enumerate(class_columns)}
        result = y_pred.copy()

        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            log.warning("DAG zawiera cykl – pomijam propagację etykiet.")
            return result

        for node in topo_order:
            if node not in col_to_idx:
                continue
            active = np.where(result[:, col_to_idx[node]] == 1)[0]
            if len(active) == 0:
                continue
            for parent in self.graph.successors(node):
                if parent in col_to_idx:
                    result[active, col_to_idx[parent]] = 1

        fixed = int(np.sum(result) - np.sum(y_pred))
        log.info(f"Propagacja DAG: naprawiono {fixed} niespójnych etykiet.")
        return result

    # ------------------------------------------------------------------
    # Metryki spójności
    # ------------------------------------------------------------------

    def count_inconsistencies(
        self,
        y_prob: np.ndarray,
        class_columns: List[str],
    ) -> int:
        """Liczy pary (próbka, krawędź) gdzie P(dziecko) > P(rodzic).

        Przydatne do monitorowania jakości kalibracji predykcji względem DAG.

        Args:
            y_prob:        Macierz prawdopodobieństw (n_samples, n_classes).
            class_columns: Lista nazw klas odpowiadających kolumnom y_prob.

        Returns:
            Liczba naruszeń monotoniczności hierarchii.
        """
        col_to_idx = {col: i for i, col in enumerate(class_columns)}
        total = 0

        for child, parent in self.graph.edges():
            if child not in col_to_idx or parent not in col_to_idx:
                continue
            total += int(np.sum(
                y_prob[:, col_to_idx[child]] > y_prob[:, col_to_idx[parent]]
            ))

        return total
