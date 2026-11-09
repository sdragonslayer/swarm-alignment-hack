"""Communication graphs and attacker placement.

Every topology returns an undirected networkx graph on nodes 0..N-1 plus the
node id of the attacker slot. In `clean` runs that slot is played by an honest
agent but is still excluded from metrics, so clean and attack runs measure the
same population of N-1 agents.
"""

from __future__ import annotations

import networkx as nx

TOPOLOGIES = ["chain", "star_hub", "star_leaf", "complete", "ring", "small_world"]


def build(topology: str, n: int, seed: int = 0) -> tuple[nx.Graph, int]:
    if n < 3:
        raise ValueError("need at least 3 agents")
    if topology == "chain":
        return nx.path_graph(n), 0  # attacker at one end
    if topology == "star_hub":
        return nx.star_graph(n - 1), 0  # node 0 is the hub
    if topology == "star_leaf":
        return nx.star_graph(n - 1), 1
    if topology == "complete":
        return nx.complete_graph(n), 0
    if topology == "ring":
        return nx.cycle_graph(n), 0
    if topology == "small_world":
        g = nx.connected_watts_strogatz_graph(n, k=min(4, n - 1), p=0.2, seed=seed)
        return g, 0
    raise ValueError(f"unknown topology {topology}")


def hop_distances(g: nx.Graph, source: int) -> dict[int, int]:
    return dict(nx.single_source_shortest_path_length(g, source))
