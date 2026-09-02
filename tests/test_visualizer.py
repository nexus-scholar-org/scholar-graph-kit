from pathlib import Path
import networkx as nx

from scholar_graph.visualizer import GraphVisualizer


def test_graph_visualizer_html(tmp_path: Path):
    G = nx.DiGraph()
    G.add_node("10.1000/1", title="Sample Paper", year=2023, citations=42)
    G.add_node("10.1000/2", title="Cited Paper", year=2020, citations=150)
    G.add_edge("10.1000/1", "10.1000/2")

    out_html = tmp_path / "test_graph.html"
    vis = GraphVisualizer(out_html)
    vis.generate_html(G)

    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "Sample Paper" in content or "10.1000/1" in content
    assert "<html" in content.lower()
