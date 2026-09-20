"""Scientometric analysis engine for citation networks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ScientometricEngine:
    """Engine for computing scientometric metrics on citation graphs."""

    def __init__(self, graph: Any = None):
        """Initialize with an optional networkx graph."""
        self.graph = graph

    def compute_hits(
        self, max_iter: int = 100, normalized: bool = True
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute HITS hubs and authorities scores."""
        import networkx as nx

        if self.graph is None or len(self.graph.nodes) == 0:
            raise ValueError("Cannot compute HITS on empty graph")

        try:
            hubs, authorities = nx.hits(
                self.graph, max_iter=max_iter, normalized=normalized
            )
        except nx.PowerIterationFailedConvergence:
            hubs, authorities = nx.hits(
                self.graph, max_iter=max_iter * 3, normalized=False
            )
            # Manual normalization using absolute values to handle negative scores
            if hubs:
                max_h = max(abs(v) for v in hubs.values()) if hubs else 1.0
                hubs = {k: v / max_h for k, v in hubs.items()}
            if authorities:
                max_a = (
                    max(abs(v) for v in authorities.values()) if authorities else 1.0
                )
                authorities = {k: v / max_a for k, v in authorities.items()}

        return hubs, authorities

    def compute_betweenness_centrality(
        self, normalized: bool = True
    ) -> dict[str, float]:
        """Compute betweenness centrality for all nodes."""
        import networkx as nx

        if self.graph is None or len(self.graph.nodes) == 0:
            raise ValueError("Cannot compute betweenness on empty graph")

        return nx.betweenness_centrality(self.graph, normalized=normalized)

    def classify_nodes_by_hits(
        self,
        hubs: dict[str, float],
        authorities: dict[str, float],
        top_k: int | None = None,
        percentile: float = 90.0,
    ) -> dict[str, list[dict]]:
        """Classify nodes into Authorities, Hubs, and Others based on HITS scores.

        Args:
            hubs: Hub scores from compute_hits().
            authorities: Authority scores from compute_hits().
            top_k: Optional limit on nodes per category.
            percentile: Percentile threshold for classification (default: 90).

        Returns:
            Dict with 'authorities', 'hubs', and 'others' lists.
        """
        import numpy as np

        result: dict[str, list[dict]] = {
            "authorities": [],
            "hubs": [],
            "others": [],
        }

        if not hubs or not authorities:
            return result

        # Compute thresholds using percentile
        all_hubs = list(hubs.values())
        all_auth = list(authorities.values())

        hub_threshold = float(np.percentile(all_hubs, percentile)) if all_hubs else 0
        auth_threshold = float(np.percentile(all_auth, percentile)) if all_auth else 0

        # Classify each node
        for node in set(list(hubs.keys()) + list(authorities.keys())):
            h_score = hubs.get(node, 0.0)
            a_score = authorities.get(node, 0.0)

            if a_score >= auth_threshold and a_score > h_score:
                result["authorities"].append(
                    {"doi": node, "score": a_score, "role": "authority"}
                )
            elif h_score >= hub_threshold and h_score > a_score:
                result["hubs"].append({"doi": node, "score": h_score, "role": "hub"})
            else:
                result["others"].append(
                    {
                        "doi": node,
                        "score": max(h_score, a_score),
                        "role": "other",
                    }
                )

        # Sort each category by score descending
        for category in result:
            result[category].sort(key=lambda x: x["score"], reverse=True)

        # Apply top_k limit if specified
        if top_k:
            for category in result:
                result[category] = result[category][:top_k]

        return result

    def build_cocitation_network(self, min_jaccard: float = 0.15) -> Any:
        """Build co-citation network using Jaccard similarity.

        Two papers A and B are co-cited if both are cited by the same paper C.
        Edge weight = Jaccard similarity of their co-citing sets.

        Args:
            min_jaccard: Minimum Jaccard similarity to create an edge.

        Returns:
            Undirected networkx.Graph with co-citation edges.
        """
        import networkx as nx

        if self.graph is None or len(self.graph.nodes) == 0:
            return nx.Graph()

        # Build forward adjacency: cites[source] = {target1, target2, ...}
        cites: dict[str, set[str]] = {}
        for source, target in self.graph.edges():
            if source not in cites:
                cites[source] = set()
            cites[source].add(target)

        # Build reverse adjacency: cited_by[target] = {source1, source2, ...}
        cited_by: dict[str, set[str]] = {}
        for source, target in self.graph.edges():
            if target not in cited_by:
                cited_by[target] = set()
            cited_by[target].add(source)

        # Find co-cited pairs: for each citing paper, pair up papers it cites
        co_citation_counts: dict[tuple[str, str], int] = {}
        for _source, targets in cites.items():
            targets_list = sorted(targets)
            for i in range(len(targets_list)):
                for j in range(i + 1, len(targets_list)):
                    pair = (targets_list[i], targets_list[j])
                    co_citation_counts[pair] = co_citation_counts.get(pair, 0) + 1

        # Build co-citation graph with Jaccard weights
        G = nx.Graph()

        # Add all nodes from original graph (including isolated ones)
        for node in self.graph.nodes():
            G.add_node(node)

        for (paper_a, paper_b), count in co_citation_counts.items():
            # Compute Jaccard similarity of their citing sets
            set_a = cited_by.get(paper_a, set())
            set_b = cited_by.get(paper_b, set())
            union = len(set_a | set_b)
            intersection = len(set_a & set_b)
            jaccard = intersection / union if union > 0 else 0

            if jaccard >= min_jaccard:
                G.add_edge(
                    paper_a,
                    paper_b,
                    weight=jaccard,
                    co_citation_count=count,
                    type="cocitation",
                )

        return G

    def build_bibliographic_coupling(self, min_overlap: int = 2) -> Any:
        """Build bibliographic coupling network.

        Two papers A and B are bibliographically coupled if they share common references.
        Edge weight = number of shared references (raw overlap).

        Args:
            min_overlap: Minimum number of shared references to create an edge.

        Returns:
            Undirected networkx.Graph with coupling edges.
        """
        import networkx as nx

        if self.graph is None or len(self.graph.nodes) == 0:
            return nx.Graph()

        # Build forward adjacency: references[source] = {target1, target2, ...}
        references: dict[str, set[str]] = {}
        for source, target in self.graph.edges():
            if source not in references:
                references[source] = set()
            references[source].add(target)

        # Build reverse adjacency: cited_by[target] = {source1, source2, ...}
        # Then invert: for each target, find all sources that cite it
        # This gives us pairs of sources that share references
        ref_to_sources: dict[str, set[str]] = {}
        for source, targets in references.items():
            for target in targets:
                if target not in ref_to_sources:
                    ref_to_sources[target] = set()
                ref_to_sources[target].add(source)

        # Find bibliographically coupled pairs
        coupling_counts: dict[tuple[str, str], int] = {}
        for _ref, sources in ref_to_sources.items():
            sources_list = sorted(sources)
            for i in range(len(sources_list)):
                for j in range(i + 1, len(sources_list)):
                    pair = (sources_list[i], sources_list[j])
                    coupling_counts[pair] = coupling_counts.get(pair, 0) + 1

        # Build coupling graph with raw overlap weights
        G = nx.Graph()
        for (paper_a, paper_b), overlap in coupling_counts.items():
            if paper_a not in G:
                G.add_node(paper_a)
            if paper_b not in G:
                G.add_node(paper_b)

            if overlap >= min_overlap:
                G.add_edge(paper_a, paper_b, weight=overlap, type="coupling")

        return G

    def build_hybrid_network(
        self,
        weight_cocite: float = 0.5,
        weight_couple: float = 0.5,
        min_weight: float = 0.1,
    ) -> Any:
        """Build hybrid co-citation + coupling network.

        Args:
            weight_cocite: Weight for co-citation component. Default: 0.5.
            weight_couple: Weight for coupling component. Default: 0.5.
            min_weight: Minimum combined weight to include edge. Default: 0.1.

        Returns:
            Undirected networkx.Graph with combined edges.
        """
        import networkx as nx

        if self.graph is None or len(self.graph.nodes) == 0:
            return nx.Graph()

        # Get co-citation and coupling networks
        cocitation = self.build_cocitation_network(min_jaccard=0.0)
        coupling = self.build_bibliographic_coupling(min_overlap=1)

        # Combine with weighted sum
        hybrid = nx.Graph()

        # Add all nodes
        for node in list(cocitation.nodes) + list(coupling.nodes):
            if node not in hybrid:
                hybrid.add_node(node)

        # Add edges with combined weights
        edge_weights: dict[tuple, float] = {}

        for u, v, data in cocitation.edges(data=True):
            weight = data.get("weight", 0)
            key = tuple(sorted([u, v]))
            edge_weights[key] = edge_weights.get(key, 0) + weight_cocite * weight

        for u, v, data in coupling.edges(data=True):
            weight = data.get("weight", 0)
            key = tuple(sorted([u, v]))
            # Normalize coupling weight by overlap coefficient
            refs_u = len(self.graph.edges(u)) if u in self.graph else 1
            refs_v = len(self.graph.edges(v)) if v in self.graph else 1
            overlap_coeff = (
                weight / min(refs_u, refs_v) if min(refs_u, refs_v) > 0 else 0
            )
            edge_weights[key] = edge_weights.get(key, 0) + weight_couple * overlap_coeff

        # Add edges with combined weights (minimum threshold)
        for (u, v), weight in edge_weights.items():
            if weight >= min_weight:
                hybrid.add_edge(u, v, weight=weight)

        return hybrid

    def detect_communities_louvain(
        self, resolution: float = 1.0, seed: int = 42
    ) -> dict[str, int]:
        """Detect communities using Louvain method.

        Args:
            resolution: Resolution parameter (higher = more communities).
            seed: Random seed for reproducibility.

        Returns:
            Dict mapping node ID to community ID (0-based).
        """
        import networkx as nx

        if self.graph is None or len(self.graph.nodes) == 0:
            return {}

        # Convert to undirected for community detection
        G_undirected = (
            self.graph.to_undirected() if self.graph.is_directed() else self.graph
        )

        # Use networkx's greedy_modularity_communities as Louvain approximation
        # (networkx doesn't have exact Louvain but greedy modularity is similar)
        from networkx.algorithms.community import greedy_modularity_communities

        communities_list = greedy_modularity_communities(
            G_undirected, resolution=resolution
        )

        # Convert list of sets to dict mapping node -> community_id
        communities: dict[str, int] = {}
        for idx, community in enumerate(communities_list):
            for node in community:
                communities[node] = idx

        return communities

    def compute_modularity(self, communities: dict[str, int]) -> float:
        """Compute modularity score for community assignment.

        Args:
            communities: Dict mapping node ID to community ID.

        Returns:
            Modularity score between -1 and 1.
        """
        import networkx as nx

        if self.graph is None or len(self.graph.nodes) == 0:
            return 0.0

        # Convert communities dict to list of sets
        community_sets: dict[int, set] = {}
        for node, comm_id in communities.items():
            if comm_id not in community_sets:
                community_sets[comm_id] = set()
            community_sets[comm_id].add(node)

        community_list = list(community_sets.values())

        if not community_list:
            return 0.0

        try:
            modularity = nx.algorithms.community.modularity(self.graph, community_list)
        except Exception:
            modularity = 0.0

        return modularity

    def enrich_graph_with_communities(self, communities: dict[str, int]) -> None:
        """Add community and group attributes to graph nodes.

        Args:
            communities: Dict mapping node ID to community ID (0-based).

        Side Effect:
            Modifies self.graph.nodes in-place, adding 'community' (0-based)
            and 'group' (1-based, for PyVis color coding) attributes.
        """
        if self.graph is None or not communities:
            return

        for node, community_id in communities.items():
            if node in self.graph:
                self.graph.nodes[node]["community"] = community_id
                self.graph.nodes[node]["group"] = community_id + 1  # 1-based for PyVis
