# Scholar Graph Kit

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Scholar Graph Kit** is a bibliometric network construction and visualization engine for scientific literature. It fetches Open Access citation and co-citation structures from OpenAlex, computes normalized **PageRank** centrality metrics to boost downstream semantic RAG retrieval, and generates interactive force-directed HTML network graphs using PyVis.

---

## Key Features

1. **Citation & Co-Citation Graph Builder (`CitationGraphBuilder`)**:
   - Asynchronously resolves references across candidates from `scholar-search-kit` screening results (`included.json`).
   - Builds directed citation graphs (`nx.DiGraph`) linking citing and cited publications.
2. **PageRank Score Computation**:
   - Calculates scale-safe normalized PageRank scores for all publications in the literature graph:
     $$\text{PageRank}(u) = \frac{1 - d}{N} + d \sum_{v \in B_u} \frac{\text{PageRank}(v)}{L(v)}$$
3. **Graph Topology & Node-Link JSON Export**:
   - Exports standard node-link JSON (`knowledge_graph.json`) ready for ingestion by `scholar-rag-kit`'s hybrid retriever.
4. **Interactive PyVis HTML Visualization (`GraphVisualizer`)**:
   - Generates standalone, self-contained interactive network maps (`knowledge_graph.html`) with customizable physics layouts and rich hover tooltips.

---

## Installation

Ensure `uv` is installed on your system.

```bash
# Clone the repository
git clone https://github.com/nexus-scholar-org/scholar-graph-kit.git
cd scholar-graph-kit

# Install dependencies in editable mode
uv pip install -e .
```

---

## Command Line Interface (CLI)

### 1. Build Citation Network from Screening Output
```bash
# Build network and export both interactive HTML visualization and PageRank JSON
uv run scholar-graph build \
  --input workspaces/my-project/literature/included.json \
  --output workspaces/my-project/literature/knowledge_graph.html \
  --json-output workspaces/my-project/literature/knowledge_graph.json
```

### 2. Build Network from Specific DOIs
```bash
uv run scholar-graph build \
  --doi 10.1038/s41586-024-0001 \
  --doi 10.1038/s41586-024-0002 \
  --output knowledge_graph.html
```

### 3. Display PageRank Rankings
```bash
uv run scholar-graph pagerank workspaces/my-project/literature/knowledge_graph.json
```

---

## Python API Usage

```python
import asyncio
from scholar_graph.builder import CitationGraphBuilder
from scholar_graph.visualizer import GraphVisualizer
from scholar_search.http_client import AcademicHttpClient

async def main():
    http_client = AcademicHttpClient(name="openalex-graph")
    builder = CitationGraphBuilder(http_client)
    
    # 1. Build directed citation graph from DOIs
    G = await builder.build_graph([
        "10.1038/s41586-024-0001",
        "10.1038/s41586-024-0002"
    ])
    
    # 2. Compute PageRank scores
    pr_scores = CitationGraphBuilder.compute_pagerank(G)
    print("Top PageRank node:", max(pr_scores, key=pr_scores.get))
    
    # 3. Export Node-Link JSON & PyVis HTML
    builder.export_json(G, "knowledge_graph.json")
    vis = GraphVisualizer("knowledge_graph.html")
    vis.generate_html(G)

asyncio.run(main())
```

---

## License

MIT License. Part of the [Nexus Scholar Suite](https://github.com/nexus-scholar-org).
