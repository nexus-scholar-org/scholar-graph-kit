import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from typer.testing import CliRunner
import networkx as nx

from scholar_graph.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    assert "--doi" in result.stdout
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--json-output" in result.stdout


@patch("scholar_graph.cli.CitationGraphBuilder.build_graph", new_callable=AsyncMock)
def test_cli_graph_build_from_file(mock_build_graph, tmp_path: Path):
    G = nx.DiGraph()
    G.add_node("10.1000/1", title="Paper One", year=2023, citations=10)
    mock_build_graph.return_value = G

    input_file = tmp_path / "included.json"
    input_file.write_text(json.dumps([{"doi": "10.1000/1"}]), encoding="utf-8")

    out_html = tmp_path / "graph.html"
    out_json = tmp_path / "graph.json"

    result = runner.invoke(
        app,
        ["build", "--input", str(input_file), "--output", str(out_html), "--json-output", str(out_json)],
    )

    debug_text = f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert result.exit_code == 0, debug_text
    assert out_html.exists()
    assert out_json.exists()
    assert "Graph built successfully" in result.stdout
