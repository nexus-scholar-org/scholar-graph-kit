from pathlib import Path

import networkx as nx

from scholar_graph.visualizer import GraphVisualizer


# ---------------------------------------------------------------------------
# Existing test (backward compat)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 1: Backward compat & constructor
# ---------------------------------------------------------------------------


def test_graph_visualizer_backward_compat(tmp_path: Path):
    """Old single-arg constructor still works and produces valid HTML."""
    G = nx.DiGraph()
    G.add_node("A", title="Paper A", citations=10)
    G.add_node("B", title="Paper B", citations=20)
    G.add_edge("A", "B")

    out = tmp_path / "compat.html"
    vis = GraphVisualizer(out)
    vis.generate_html(G)
    assert out.exists()
    assert "<html" in out.read_text(encoding="utf-8").lower()


def test_graph_visualizer_constructor_params(tmp_path: Path):
    """New constructor parameters are stored as instance attributes."""
    vis = GraphVisualizer(
        tmp_path / "out.html",
        physics_enabled=False,
        height="600px",
        width="80%",
        bgcolor="#000000",
        font_color="#ffffff",
    )
    assert vis.physics_enabled is False
    assert vis.height == "600px"
    assert vis.width == "80%"
    assert vis.bgcolor == "#000000"
    assert vis.font_color == "#ffffff"


# ---------------------------------------------------------------------------
# Task 2: Helper methods
# ---------------------------------------------------------------------------


def _make_graph():
    G = nx.DiGraph()
    G.add_node("A", title="A", year=2020, citations=100, community="ML")
    G.add_node("B", title="B", year=2021, citations=50, community="NLP")
    G.add_node("C", title="C", year=2022, citations=10, community="ML")
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    return G


def test_compute_node_sizes_normalization(tmp_path: Path):
    G = _make_graph()
    vis = GraphVisualizer(tmp_path / "x.html")
    sizes = vis._compute_node_sizes(G, "citations")
    assert all(isinstance(v, int) for v in sizes.values())
    assert all(10 <= v <= 30 for v in sizes.values())
    # A=100, B=50, C=10 => A largest, C smallest
    assert sizes["A"] > sizes["C"]


def test_compute_node_sizes_empty_graph(tmp_path: Path):
    G = nx.DiGraph()
    vis = GraphVisualizer(tmp_path / "x.html")
    sizes = vis._compute_node_sizes(G)
    assert sizes == {}


def test_compute_node_sizes_uniform_values(tmp_path: Path):
    G = nx.DiGraph()
    G.add_node("X", citations=5)
    G.add_node("Y", citations=5)
    vis = GraphVisualizer(tmp_path / "x.html")
    sizes = vis._compute_node_sizes(G)
    assert sizes["X"] == sizes["Y"] == 10  # min_size default


def test_compute_node_colors_deterministic(tmp_path: Path):
    G = _make_graph()
    vis = GraphVisualizer(tmp_path / "x.html")
    colors1 = vis._compute_node_colors(G, "community")
    colors2 = vis._compute_node_colors(G, "community")
    assert colors1 == colors2
    # ML nodes should share a color
    assert colors1["A"] == colors1["C"]
    # NLP node different from ML
    assert colors1["B"] != colors1["A"]


def test_build_tooltip_content(tmp_path: Path):
    vis = GraphVisualizer(tmp_path / "x.html")
    tooltip = vis._build_tooltip(
        "10.1000/test", {"title": "My Paper", "year": 2024, "citations": 42}
    )
    assert "<b>" in tooltip
    assert "DOI" in tooltip
    assert "10.1000/test" in tooltip
    assert "My Paper" in tooltip
    assert "2024" in tooltip


# ---------------------------------------------------------------------------
# Task 3 & 4: generate_html + CSS injection
# ---------------------------------------------------------------------------


def test_graph_visualizer_community_coloring(tmp_path: Path):
    G = _make_graph()
    out = tmp_path / "community.html"
    vis = GraphVisualizer(out)
    vis.generate_html(G, node_color_attr="community")
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content.lower()


def test_graph_visualizer_no_physics(tmp_path: Path):
    G = _make_graph()
    out = tmp_path / "nophys.html"
    vis = GraphVisualizer(out, physics_enabled=False)
    vis.generate_html(G)
    assert out.exists()


def test_inject_custom_css(tmp_path: Path):
    G = _make_graph()
    out = tmp_path / "styled.html"
    vis = GraphVisualizer(out)
    vis.generate_html(G)
    content = out.read_text(encoding="utf-8")
    assert "<style>" in content
    assert "#mynetwork" in content


def test_generate_html_returns_path(tmp_path: Path):
    G = nx.DiGraph()
    G.add_node("X", title="X", citations=1)
    out = tmp_path / "ret.html"
    vis = GraphVisualizer(out)
    result = vis.generate_html(G)
    assert result == out


# ---------------------------------------------------------------------------
# Task 5: CLI command
# ---------------------------------------------------------------------------


def test_visualize_cli_command(tmp_path: Path):
    """Integration test for the visualize CLI command."""
    from typer.testing import CliRunner

    from scholar_graph.cli import app

    runner = CliRunner()

    # Build a small graph JSON fixture
    G = nx.DiGraph()
    G.add_node("10.1000/1", title="Test Paper", year=2023, citations=42)
    G.add_node("10.1000/2", title="Other Paper", year=2020, citations=10)
    G.add_edge("10.1000/1", "10.1000/2")

    import json

    graph_json = tmp_path / "graph.json"
    data = nx.node_link_data(G)
    graph_json.write_text(json.dumps(data), encoding="utf-8")

    out_html = tmp_path / "output.html"
    result = runner.invoke(
        app,
        ["visualize", str(graph_json), "-o", str(out_html)],
    )
    assert result.exit_code == 0, result.stdout
    assert out_html.exists()


def test_visualize_cli_help():
    """visualize command appears in --help."""
    from typer.testing import CliRunner

    from scholar_graph.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["visualize", "--help"])
    assert result.exit_code == 0
    assert (
        "graph JSON" in result.stdout.lower()
        or "visualization" in result.stdout.lower()
    )
