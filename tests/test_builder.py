import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
import networkx as nx

from scholar_graph.builder import CitationGraphBuilder
from scholar_search.http_client import AcademicHttpClient


@pytest.mark.asyncio
async def test_citation_graph_builder(tmp_path: Path):
    mock_http_client = MagicMock(spec=AcademicHttpClient)

    async def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if "10.1000/1" in url:
            mock_resp.json.return_value = {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1000/1",
                "title": "Paper One",
                "publication_year": 2023,
                "cited_by_count": 50,
                "referenced_works": ["https://openalex.org/W2"],
            }
        elif "10.1000/2" in url:
            mock_resp.json.return_value = {
                "id": "https://openalex.org/W2",
                "doi": "https://doi.org/10.1000/2",
                "title": "Paper Two",
                "publication_year": 2022,
                "cited_by_count": 120,
                "referenced_works": [],
            }
        else:
            mock_resp.status_code = 404
            mock_resp.json.return_value = {}
        return mock_resp

    mock_http_client.get = AsyncMock(side_effect=mock_get)

    builder = CitationGraphBuilder(mock_http_client)
    G = await builder.build_graph(["10.1000/1", "10.1000/2"])

    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes) == 2
    assert ("10.1000/1", "10.1000/2") in G.edges

    pr = CitationGraphBuilder.compute_pagerank(G)
    assert "10.1000/1" in pr
    assert "10.1000/2" in pr

    json_path = tmp_path / "graph.json"
    exported = CitationGraphBuilder.export_json(G, json_path)
    assert exported.exists()
