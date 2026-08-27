from dataclasses import dataclass, field

@dataclass
class GraphNode:
    doi: str
    title: str = "Unknown Title"
    year: int | None = None
    citations: int = 0
    group: int = 1  # For coloring clusters in PyVis

@dataclass
class GraphEdge:
    source: str
    target: str
