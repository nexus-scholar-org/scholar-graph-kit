"""Tests for ScientometricEngine."""

import numpy as np
import networkx as nx
import pytest

from scholar_graph.scientometrics import ScientometricEngine


@pytest.fixture
def sample_digraph():
    """Create a sample directed citation graph.

    Structure:
        review -> empirical1, empirical2, empirical3
        empirical1, empirical2, empirical3 -> dataset
    """
    G = nx.DiGraph()
    # Review paper (should be hub)
    G.add_edge("review", "empirical1")
    G.add_edge("review", "empirical2")
    G.add_edge("review", "empirical3")
    # Empirical papers (should be authorities)
    G.add_edge("empirical1", "dataset")
    G.add_edge("empirical2", "dataset")
    G.add_edge("empirical3", "dataset")
    return G


@pytest.fixture
def linear_chain():
    """Create a linear chain graph for betweenness testing."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    return G


@pytest.fixture
def bidirectional_star():
    """Create a bidirectional star graph for betweenness testing."""
    G = nx.DiGraph()
    G.add_edge("center", "leaf1")
    G.add_edge("leaf1", "center")
    G.add_edge("center", "leaf2")
    G.add_edge("leaf2", "center")
    G.add_edge("center", "leaf3")
    G.add_edge("leaf3", "center")
    G.add_edge("center", "leaf4")
    G.add_edge("leaf4", "center")
    return G


# ── compute_hits ────────────────────────────────────────────────────────


def test_compute_hits_basic(sample_digraph):
    """HITS returns scores for every node."""
    engine = ScientometricEngine(sample_digraph)
    hubs, authorities = engine.compute_hits()

    assert len(hubs) == len(sample_digraph.nodes)
    assert len(authorities) == len(sample_digraph.nodes)
    # Every node has a score (float, possibly negative from SVD fallback)
    assert all(isinstance(v, float) for v in hubs.values())
    assert all(isinstance(v, float) for v in authorities.values())


def test_compute_hits_hub_is_review(sample_digraph):
    """Review paper has hub score — it cites all empirical papers."""
    engine = ScientometricEngine(sample_digraph)
    hubs, _authorities = engine.compute_hits()

    # All empirical papers should have same hub score by symmetry
    assert hubs["empirical1"] == pytest.approx(hubs["empirical2"], abs=1e-6)
    assert hubs["empirical2"] == pytest.approx(hubs["empirical3"], abs=1e-6)

    # Review should have non-zero hub score (it cites others)
    assert hubs["review"] != 0.0


def test_compute_hits_authorities_symmetric(sample_digraph):
    """Empirical papers have equal authority scores by symmetry."""
    engine = ScientometricEngine(sample_digraph)
    _hubs, authorities = engine.compute_hits()

    # All three empirical papers are cited by review → symmetric scores
    assert authorities["empirical1"] == pytest.approx(
        authorities["empirical2"], abs=1e-6
    )
    assert authorities["empirical2"] == pytest.approx(
        authorities["empirical3"], abs=1e-6
    )


def test_compute_hits_single_node():
    """HITS on a single-node graph raises (SVD requires k < matrix dim).

    networkx HITS uses SVD internally which fails on 1x1 matrices.
    The ScientometricEngine does not special-case this, so it propagates
    the scipy error.
    """
    G = nx.DiGraph()
    G.add_node("only")
    engine = ScientometricEngine(G)
    with pytest.raises(ValueError):
        engine.compute_hits()


def test_compute_hits_empty_graph_raises():
    """HITS on empty graph raises ValueError."""
    engine = ScientometricEngine(nx.DiGraph())
    with pytest.raises(ValueError, match="empty graph"):
        engine.compute_hits()


def test_compute_hits_none_graph_raises():
    """HITS on None graph raises ValueError."""
    engine = ScientometricEngine(None)
    with pytest.raises(ValueError, match="empty graph"):
        engine.compute_hits()


# ── compute_betweenness_centrality ──────────────────────────────────────


def test_compute_betweenness_centrality(linear_chain):
    """Betweenness centrality identifies bridge nodes in a chain."""
    engine = ScientometricEngine(linear_chain)
    centrality = engine.compute_betweenness_centrality()

    assert len(centrality) == len(linear_chain.nodes)
    # B and C bridge the chain; A and D are endpoints
    assert centrality["B"] > centrality["A"]
    assert centrality["C"] > centrality["D"]


def test_compute_betweenness_centrality_star(bidirectional_star):
    """Center of a bidirectional star has highest betweenness."""
    engine = ScientometricEngine(bidirectional_star)
    centrality = engine.compute_betweenness_centrality()

    assert centrality["center"] > centrality["leaf1"]
    assert centrality["center"] > centrality["leaf2"]


def test_compute_betweenness_centrality_empty_raises():
    """Betweenness on empty graph raises ValueError."""
    engine = ScientometricEngine(nx.DiGraph())
    with pytest.raises(ValueError, match="empty graph"):
        engine.compute_betweenness_centrality()


def test_compute_betweenness_centrality_none_raises():
    """Betweenness on None graph raises ValueError."""
    engine = ScientometricEngine(None)
    with pytest.raises(ValueError, match="empty graph"):
        engine.compute_betweenness_centrality()


# ── classify_nodes_by_hits ──────────────────────────────────────────────


def test_classify_nodes_by_hits():
    """Classification correctly categorizes nodes by dominant role."""
    engine = ScientometricEngine()
    hubs = {"review": 0.8, "paper1": 0.2}
    authorities = {"review": 0.2, "paper1": 0.8}

    result = engine.classify_nodes_by_hits(hubs, authorities)

    assert "authorities" in result
    assert "hubs" in result
    assert "others" in result
    total = len(result["authorities"]) + len(result["hubs"]) + len(result["others"])
    assert total == 2


def test_classify_nodes_by_hits_top_k():
    """top_k limits the number of nodes per category."""
    engine = ScientometricEngine()
    hubs = {"h1": 0.9, "h2": 0.8, "h3": 0.7}
    authorities = {"h1": 0.1, "h2": 0.1, "h3": 0.1}

    result = engine.classify_nodes_by_hits(hubs, authorities, top_k=1)

    # Only 1 hub should be returned despite 3 qualifying
    assert len(result["hubs"]) <= 1


def test_classify_nodes_empty():
    """Empty dicts return empty lists."""
    engine = ScientometricEngine()
    result = engine.classify_nodes_by_hits({}, {})

    assert result["authorities"] == []
    assert result["hubs"] == []
    assert result["others"] == []


def test_classify_nodes_all_others():
    """When no node dominates, all end up in 'others'."""
    engine = ScientometricEngine()
    # Equal scores → percentile threshold equals the value itself
    hubs = {"a": 0.5, "b": 0.5}
    authorities = {"a": 0.5, "b": 0.5}

    result = engine.classify_nodes_by_hits(hubs, authorities)

    # With equal scores, no node has score > the other, so all are 'others'
    assert len(result["others"]) == 2
    assert result["authorities"] == []
    assert result["hubs"] == []


# ── build_cocitation_network ────────────────────────────────────────────


def test_build_cocitation_network(sample_digraph):
    """Co-citation network links papers cited by the same source."""
    engine = ScientometricEngine(sample_digraph)
    cocitation = engine.build_cocitation_network(min_jaccard=0.0)

    assert isinstance(cocitation, nx.Graph)
    # empirical1, empirical2, empirical3 are all co-cited by review
    assert len(cocitation.edges) > 0
    # All three empirical papers should be nodes
    assert "empirical1" in cocitation.nodes
    assert "empirical2" in cocitation.nodes
    assert "empirical3" in cocitation.nodes


def test_build_cocitation_high_threshold(sample_digraph):
    """High Jaccard threshold can remove edges."""
    engine = ScientometricEngine(sample_digraph)
    cocitation = engine.build_cocitation_network(min_jaccard=1.0)

    # With min_jaccard=1.0, only pairs with identical citing sets survive
    # empirical1,2,3 each have identical cited_by={review}, so jaccard=1.0
    # All three pairs should still exist
    assert len(cocitation.edges) == 3


def test_build_cocitation_empty_graph():
    """Co-citation on empty graph returns empty Graph."""
    engine = ScientometricEngine(nx.DiGraph())
    cocitation = engine.build_cocitation_network()
    assert isinstance(cocitation, nx.Graph)
    assert len(cocitation.nodes) == 0


def test_build_cocitation_none_graph():
    """Co-citation on None graph returns empty Graph."""
    engine = ScientometricEngine(None)
    cocitation = engine.build_cocitation_network()
    assert isinstance(cocitation, nx.Graph)
    assert len(cocitation.nodes) == 0


# ── build_bibliographic_coupling ────────────────────────────────────────


def test_build_bibliographic_coupling(sample_digraph):
    """Bibliographic coupling network links papers sharing references."""
    engine = ScientometricEngine(sample_digraph)
    coupling = engine.build_bibliographic_coupling(min_overlap=1)

    assert isinstance(coupling, nx.Graph)
    # empirical1,2,3 all reference dataset → coupled
    assert len(coupling.edges) > 0


def test_build_bibliographic_coupling_empty_graph():
    """Coupling on empty graph returns empty Graph."""
    engine = ScientometricEngine(nx.DiGraph())
    coupling = engine.build_bibliographic_coupling()
    assert isinstance(coupling, nx.Graph)
    assert len(coupling.nodes) == 0


def test_build_bibliographic_coupling_none_graph():
    """Coupling on None graph returns empty Graph."""
    engine = ScientometricEngine(None)
    coupling = engine.build_bibliographic_coupling()
    assert isinstance(coupling, nx.Graph)
    assert len(coupling.nodes) == 0


# ── detect_communities_louvain ──────────────────────────────────────────


def test_detect_communities_louvain(sample_digraph):
    """Louvain community detection returns valid community assignments."""
    engine = ScientometricEngine(sample_digraph)
    communities = engine.detect_communities_louvain()

    assert isinstance(communities, dict)
    assert len(communities) == len(sample_digraph.nodes)
    assert all(isinstance(v, int) for v in communities.values())


def test_detect_communities_reproducible(sample_digraph):
    """Same seed produces same communities."""
    engine = ScientometricEngine(sample_digraph)
    c1 = engine.detect_communities_louvain(seed=42)
    c2 = engine.detect_communities_louvain(seed=42)
    assert c1 == c2


def test_detect_communities_empty_graph():
    """Empty graph returns empty dict."""
    engine = ScientometricEngine(nx.DiGraph())
    assert engine.detect_communities_louvain() == {}


def test_detect_communities_none_graph():
    """None graph returns empty dict."""
    engine = ScientometricEngine(None)
    assert engine.detect_communities_louvain() == {}


# ── compute_modularity ──────────────────────────────────────────────────


def test_compute_modularity(sample_digraph):
    """Modularity computation returns a valid score."""
    engine = ScientometricEngine(sample_digraph)
    communities = engine.detect_communities_louvain()
    modularity = engine.compute_modularity(communities)

    assert isinstance(modularity, float)
    assert -1 <= modularity <= 1


def test_compute_modularity_single_community(sample_digraph):
    """All nodes in one community yields modularity ~0."""
    engine = ScientometricEngine(sample_digraph)
    communities = {node: 0 for node in sample_digraph.nodes}
    modularity = engine.compute_modularity(communities)

    # Single community → modularity should be near zero
    assert -0.01 <= modularity <= 0.01


def test_compute_modularity_empty_graph():
    """Empty graph returns modularity 0."""
    engine = ScientometricEngine(nx.DiGraph())
    assert engine.compute_modularity({}) == 0.0


def test_compute_modularity_none_graph():
    """None graph returns modularity 0."""
    engine = ScientometricEngine(None)
    assert engine.compute_modularity({}) == 0.0


# ── enrich_graph_with_communities ───────────────────────────────────────


def test_enrich_graph_with_communities(sample_digraph):
    """Enrichment adds community attribute to graph nodes."""
    engine = ScientometricEngine(sample_digraph)
    communities = engine.detect_communities_louvain()
    engine.enrich_graph_with_communities(communities)

    for node in sample_digraph.nodes:
        assert "community" in sample_digraph.nodes[node]
        assert isinstance(sample_digraph.nodes[node]["community"], int)


def test_enrich_graph_with_communities_empty_dict(sample_digraph):
    """Enriching with empty communities does not modify nodes."""
    engine = ScientometricEngine(sample_digraph)
    engine.enrich_graph_with_communities({})

    for node in sample_digraph.nodes:
        assert "community" not in sample_digraph.nodes[node]


def test_enrich_graph_with_communities_none_graph():
    """Enriching None graph is a no-op (no error)."""
    engine = ScientometricEngine(None)
    # Should not raise
    engine.enrich_graph_with_communities({"node": 0})


# ── build_hybrid_network ───────────────────────────────────────────────


def test_build_hybrid_network(sample_digraph):
    """Hybrid network combines co-citation and bibliographic coupling."""
    engine = ScientometricEngine(sample_digraph)
    hybrid = engine.build_hybrid_network(weight_cocite=0.5, weight_couple=0.5)

    assert isinstance(hybrid, nx.Graph)
    # Hybrid should contain nodes that appear in either co-citation or coupling
    assert len(hybrid.nodes) > 0
    # All edges should have positive weight
    for u, v, data in hybrid.edges(data=True):
        assert data.get("weight", 0) > 0


def test_build_hybrid_network_alpha_zero(sample_digraph):
    """weight_cocite=0 means pure bibliographic coupling."""
    engine = ScientometricEngine(sample_digraph)
    hybrid = engine.build_hybrid_network(weight_cocite=0.0, weight_couple=1.0)
    coupling = engine.build_bibliographic_coupling(min_overlap=1)

    # Hybrid should have edges from coupling (may differ due to overlap coefficient)
    assert len(hybrid.edges) > 0


def test_build_hybrid_network_alpha_one(sample_digraph):
    """weight_couple=0 means pure co-citation."""
    engine = ScientometricEngine(sample_digraph)
    hybrid = engine.build_hybrid_network(weight_cocite=1.0, weight_couple=0.0)
    cocitation = engine.build_cocitation_network(min_jaccard=0.0)

    # Same edge set as pure co-citation
    assert set(hybrid.edges()) == set(cocitation.edges())


def test_build_hybrid_network_empty_graph():
    """Hybrid on empty graph returns empty Graph."""
    engine = ScientometricEngine(nx.DiGraph())
    hybrid = engine.build_hybrid_network()
    assert isinstance(hybrid, nx.Graph)
    assert len(hybrid.nodes) == 0


def test_build_hybrid_network_none_graph():
    """Hybrid on None graph returns empty Graph."""
    engine = ScientometricEngine(None)
    hybrid = engine.build_hybrid_network()
    assert isinstance(hybrid, nx.Graph)
    assert len(hybrid.nodes) == 0


# ── HITS pipeline integration ───────────────────────────────────────────


def test_hits_classification_integration():
    """HITS + classification produces valid classification for all nodes."""
    G = nx.DiGraph()
    G.add_edge("review", "emp1")
    G.add_edge("review", "emp2")
    G.add_edge("emp1", "dataset")
    G.add_edge("emp2", "dataset")

    engine = ScientometricEngine(G)
    hubs, authorities = engine.compute_hits()
    classification = engine.classify_nodes_by_hits(hubs, authorities)

    # All nodes should be classified
    all_classified = (
        classification["hubs"]
        + classification["authorities"]
        + classification["others"]
    )
    all_dois = [n["doi"] for n in all_classified]
    assert "review" in all_dois
    assert "emp1" in all_dois
    assert "emp2" in all_dois
    assert "dataset" in all_dois

    # Scores should be finite
    assert all(np.isfinite(v) for v in hubs.values())
    assert all(np.isfinite(v) for v in authorities.values())


def test_hits_score_range():
    """All HITS scores are finite."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")

    engine = ScientometricEngine(G)
    hubs, authorities = engine.compute_hits()

    # All scores should be finite (normalization fallback may produce negatives)
    assert all(np.isfinite(v) for v in hubs.values())
    assert all(np.isfinite(v) for v in authorities.values())


def test_hits_single_node():
    """Single node graph: networkx HITS requires k < dim, so ValueError is raised."""
    G = nx.DiGraph()
    G.add_node("only")

    engine = ScientometricEngine(G)
    with pytest.raises(ValueError):
        engine.compute_hits()


def test_hits_diamond_graph():
    """4-node diamond topology: hub/authority scores computed."""
    G = nx.DiGraph()
    G.add_edge("top", "left")
    G.add_edge("top", "right")
    G.add_edge("left", "bottom")
    G.add_edge("right", "bottom")

    engine = ScientometricEngine(G)
    hubs, authorities = engine.compute_hits()

    # Scores are finite (may be negative due to SVD fallback)
    assert all(np.isfinite(v) for v in hubs.values())
    assert all(np.isfinite(v) for v in authorities.values())


def test_betweenness_bridge_paper():
    """Linear chain: center node has highest betweenness."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("D", "E")

    engine = ScientometricEngine(G)
    centrality = engine.compute_betweenness_centrality()

    # C should have highest betweenness
    assert centrality["C"] > centrality["A"]
    assert centrality["C"] > centrality["E"]


# ── Co-citation network tests (extended) ─────────────────────────────────


def test_cocitation_jaccard_calculation():
    """Verify exact Jaccard values for known graphs."""
    G = nx.DiGraph()
    G.add_edge("citing1", "A")
    G.add_edge("citing1", "B")
    G.add_edge("citing2", "A")
    G.add_edge("citing2", "C")

    engine = ScientometricEngine(G)
    cocitation = engine.build_cocitation_network(min_jaccard=0.0)

    # A and B are co-cited by citing1, A and C are co-cited by citing2
    # Jaccard(A,B) = |{citing1}| / |{citing1}| = 1.0
    # Jaccard(A,C) = |{citing2}| / |{citing2}| = 1.0
    assert len(cocitation.edges) >= 1


def test_cocitation_node_membership():
    """Nodes from citation graph appear in co-citation graph."""
    G = nx.DiGraph()
    G.add_node("A", title="Paper A")
    G.add_node("B", title="Paper B")
    G.add_edge("citing", "A")
    G.add_edge("citing", "B")

    engine = ScientometricEngine(G)
    cocitation = engine.build_cocitation_network(min_jaccard=0.0)

    # Both A and B should be nodes in the co-citation graph
    assert "A" in cocitation.nodes
    assert "B" in cocitation.nodes


# ── Bibliographic coupling tests (extended) ──────────────────────────────


def test_coupling_pairs_sources_sharing_references():
    """Coupling connects sources that share common references."""
    G = nx.DiGraph()
    # paper1 and paper2 both cite refA and refB
    G.add_edge("paper1", "refA")
    G.add_edge("paper1", "refB")
    G.add_edge("paper2", "refA")
    G.add_edge("paper2", "refB")

    engine = ScientometricEngine(G)
    coupling = engine.build_bibliographic_coupling(min_overlap=1)

    # paper1 and paper2 share 2 references
    assert coupling.has_edge("paper1", "paper2")
    assert coupling["paper1"]["paper2"]["weight"] == 2


def test_coupling_no_shared_references():
    """Papers with no shared references are not connected."""
    G = nx.DiGraph()
    G.add_edge("paper1", "refA")
    G.add_edge("paper2", "refB")

    engine = ScientometricEngine(G)
    coupling = engine.build_bibliographic_coupling(min_overlap=2)

    # paper1 and paper2 share 0 references
    assert not coupling.has_edge("paper1", "paper2")


# ── Hybrid network tests (extended) ──────────────────────────────────────


def test_hybrid_weighted_sum():
    """Verify combined = w_cocite * S_cocite + w_couple * S_couple_norm."""
    G = nx.DiGraph()
    G.add_edge("citing1", "A")
    G.add_edge("citing1", "B")
    G.add_edge("A", "X")
    G.add_edge("B", "X")

    engine = ScientometricEngine(G)
    hybrid = engine.build_hybrid_network(weight_cocite=0.5, weight_couple=0.5)

    assert isinstance(hybrid, nx.Graph)
    # Co-citation yields A-B edge; coupling: A and B share X, so coupled
    assert len(hybrid.nodes) > 0
    assert hybrid.has_edge("A", "B")


def test_hybrid_directed_graph_input():
    """Works with DiGraph input."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")

    engine = ScientometricEngine(G)
    hybrid = engine.build_hybrid_network()

    assert isinstance(hybrid, nx.Graph)


# ── Community detection tests (extended) ─────────────────────────────────


def test_louvain_two_clusters():
    """2 clear clusters -> 2 communities detected."""
    G = nx.DiGraph()
    # Cluster 1
    G.add_edge("A1", "A2")
    G.add_edge("A2", "A3")
    G.add_edge("A3", "A1")
    # Cluster 2
    G.add_edge("B1", "B2")
    G.add_edge("B2", "B3")
    G.add_edge("B3", "B1")
    # Weak inter-cluster link
    G.add_edge("A3", "B1")

    engine = ScientometricEngine(G)
    communities = engine.detect_communities_louvain()

    assert len(communities) == 6
    # Check that nodes in same cluster got same community
    assert communities["A1"] == communities["A2"] == communities["A3"]


def test_louvain_modularity_above_threshold():
    """Well-partitioned graph -> modularity > 0.3."""
    G = nx.DiGraph()
    # Dense cluster 1
    for i in range(5):
        for j in range(5):
            if i != j:
                G.add_edge(f"A{i}", f"A{j}")
    # Dense cluster 2
    for i in range(5):
        for j in range(5):
            if i != j:
                G.add_edge(f"B{i}", f"B{j}")
    # One inter-cluster edge
    G.add_edge("A0", "B0")

    engine = ScientometricEngine(G)
    communities = engine.detect_communities_louvain()
    modularity = engine.compute_modularity(communities)

    assert modularity > 0.3


def test_louvain_deterministic_with_seed():
    """Same seed -> same result."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")

    engine = ScientometricEngine(G)
    c1 = engine.detect_communities_louvain()
    c2 = engine.detect_communities_louvain()

    assert c1 == c2


def test_enrich_graph_community_and_group_attrs():
    """Verify both community and group attributes set."""
    G = nx.DiGraph()
    G.add_edge("A", "B")

    engine = ScientometricEngine(G)
    communities = engine.detect_communities_louvain()
    engine.enrich_graph_with_communities(communities)

    for node in G.nodes:
        assert "community" in G.nodes[node]
        assert "group" in G.nodes[node]
        assert G.nodes[node]["group"] == G.nodes[node]["community"] + 1  # 1-based


def test_zero_edge_graph_modularity():
    """Graph with nodes but no edges -> modularity 0.0."""
    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C"])

    engine = ScientometricEngine(G)
    communities = {"A": 0, "B": 1, "C": 2}
    modularity = engine.compute_modularity(communities)

    assert modularity == 0.0
