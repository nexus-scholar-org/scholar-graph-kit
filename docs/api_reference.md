# Scholar Graph Kit: API Reference

This document provides the API contracts for the core components of `scholar-graph-kit`.

## `GraphBuilder`
Fetches citation data from OpenAlex and constructs a `GraphData` object.

```python
from scholar_graph.builder import GraphBuilder

builder = GraphBuilder(provider="openalex")

# Expand forward (citations), backward (references), or both
graph_data = builder.build_graph(seed_dois=["10.1038/nature14539"], direction="both")
```

## `NetworkAnalyzer`
Analyzes the graph topology using NetworkX.

```python
from scholar_graph.analyzer import NetworkAnalyzer

analyzer = NetworkAnalyzer(graph_data)
scores = analyzer.calculate_centrality()
# Returns dict: {"DOI": {"pagerank": 0.05, "in_degree": 10}}
```

## `GraphVisualizer`
Exports the NetworkX graph to an interactive HTML map using PyVis.

```python
from scholar_graph.visualizer import GraphVisualizer
from pathlib import Path

visualizer = GraphVisualizer(analyzer)
visualizer.export_html(Path("network.html"))
```

## Models
Strictly validated data structures using Pydantic v2.

- `GraphData`: Holds nodes (dictionary) and edges (list).
- `NodeMetadata`: Represents a paper (DOI, title, year, citations).
- `GraphEdge`: Represents a citation link (source -> target).
