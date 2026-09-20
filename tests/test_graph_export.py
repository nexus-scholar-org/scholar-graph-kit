"""Tests for GEXF and GraphML exporters."""

import pytest
import networkx as nx
from pathlib import Path

from scholar_graph.builder import CitationGraphBuilder


@pytest.fixture
def sample_graph():
    """Create a sample directed graph."""
    G = nx.DiGraph()
    G.add_edge("paper1", "paper2", weight=1.0)
    G.add_edge("paper2", "paper3", weight=0.8)
    G.add_node("paper1", title="Paper 1", pagerank=0.4)
    G.add_node("paper2", title="Paper 2", pagerank=0.35)
    G.add_node("paper3", title="Paper 3", pagerank=0.25)
    return G


def test_export_gexf_creates_file(sample_graph, tmp_path):
    """GEXF export creates output file."""
    output = tmp_path / "test.gexf"
    result = CitationGraphBuilder.export_gexf(sample_graph, output)

    assert result.exists()
    assert result.suffix == ".gexf"


def test_export_gexf_enforces_suffix(sample_graph, tmp_path):
    """GEXF export enforces .gexf suffix."""
    output = tmp_path / "test.txt"
    result = CitationGraphBuilder.export_gexf(sample_graph, output)

    assert result.suffix == ".gexf"


def test_export_gexf_content(sample_graph, tmp_path):
    """GEXF output has valid XML structure."""
    output = tmp_path / "test.gexf"
    result = CitationGraphBuilder.export_gexf(sample_graph, output)

    content = result.read_text(encoding="utf-8")
    assert "<?xml" in content
    assert "<gexf" in content
    assert "paper1" in content
    assert "paper2" in content


def test_export_graphml_creates_file(sample_graph, tmp_path):
    """GraphML export creates output file."""
    output = tmp_path / "test.graphml"
    result = CitationGraphBuilder.export_graphml(sample_graph, output)

    assert result.exists()
    assert result.suffix == ".graphml"


def test_export_graphml_enforces_suffix(sample_graph, tmp_path):
    """GraphML export enforces .graphml suffix."""
    output = tmp_path / "test.xml"
    result = CitationGraphBuilder.export_graphml(sample_graph, output)

    assert result.suffix == ".graphml"


def test_export_graphml_content(sample_graph, tmp_path):
    """GraphML output has valid XML structure."""
    output = tmp_path / "test.graphml"
    result = CitationGraphBuilder.export_graphml(sample_graph, output)

    content = result.read_text(encoding="utf-8")
    assert "<?xml" in content
    assert "<graphml" in content
    assert "paper1" in content


def test_export_gexf_auto_pagerank(sample_graph, tmp_path):
    """GEXF export adds pagerank if missing."""
    G = nx.DiGraph()
    G.add_edge("A", "B")

    output = tmp_path / "test.gexf"
    result = CitationGraphBuilder.export_gexf(G, output)

    content = result.read_text(encoding="utf-8")
    assert "pagerank" in content


def test_export_gexf_creates_dirs(sample_graph, tmp_path):
    """GEXF export creates parent directories."""
    output = tmp_path / "subdir" / "test.gexf"
    result = CitationGraphBuilder.export_gexf(sample_graph, output)

    assert result.exists()
