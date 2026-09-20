from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx


class GraphVisualizer:
    """Enhanced PyVis-based citation graph visualizer with community colors,
    configurable physics, custom CSS injection, and rich HTML tooltips."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        physics_enabled: bool = True,
        height: str = "800px",
        width: str = "100%",
        bgcolor: str = "#ffffff",
        font_color: str = "#333333",
    ):
        self.output_path = Path(output_path)
        self.physics_enabled = physics_enabled
        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _COLOR_PALETTE: list[str] = [
        "#e6194b",
        "#3cb44b",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#42d4f4",
        "#f032e6",
        "#bfef45",
        "#fabed4",
        "#469990",
        "#dcbeff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#a9a9a9",
        "#e6beff",
    ]

    def _compute_node_sizes(
        self,
        G: nx.DiGraph,
        attr: str = "citations",
        min_size: int = 10,
        max_size: int = 30,
    ) -> dict[str, int]:
        """Normalize *attr* values across nodes to [min_size, max_size].

        Returns uniform ``min_size`` when graph is empty or all values are equal.
        """
        values: dict[str, float] = {}
        for node, data in G.nodes(data=True):
            val = data.get(attr, 0)
            values[node] = float(val) if val is not None else 0.0

        if not values:
            return {}

        vals = list(values.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo

        if span == 0:
            return {n: min_size for n in values}

        result: dict[str, int] = {}
        for n, v in values.items():
            norm = (v - lo) / span
            result[n] = int(min_size + norm * (max_size - min_size))
        return result

    def _compute_node_colors(
        self,
        G: nx.DiGraph,
        attr: str = "community",
    ) -> dict[str, str]:
        """Map unique values of *attr* to a 20-color palette using modular indexing.

        Nodes missing the attribute get a default grey color.
        """
        unique_vals: list[str] = []
        seen: set[str] = set()
        for _node, data in G.nodes(data=True):
            v = str(data.get(attr, ""))
            if v and v not in seen:
                seen.add(v)
                unique_vals.append(v)

        mapping: dict[str, str] = {
            v: self._COLOR_PALETTE[i % len(self._COLOR_PALETTE)]
            for i, v in enumerate(unique_vals)
        }

        default_color = "#cccccc"
        result: dict[str, str] = {}
        for node, data in G.nodes(data=True):
            v = str(data.get(attr, ""))
            result[node] = mapping.get(v, default_color)
        return result

    def _build_tooltip(self, node: str, attrs: dict[str, Any]) -> str:
        """Build an HTML tooltip string for a single node."""
        parts: list[str] = []
        fields = [
            ("DOI", node),
            ("Title", attrs.get("title", "N/A")),
            ("Year", attrs.get("year", "N/A")),
            ("Citations", attrs.get("citations", 0)),
            ("Community", attrs.get("community", "N/A")),
            ("PageRank", attrs.get("pagerank", "N/A")),
        ]
        for label, value in fields:
            parts.append(f"<b>{label}:</b> {value}")
        return "<br>".join(parts)

    def _inject_custom_css(self, html_path: Path) -> None:
        """Inject a ``<style>`` block before ``</head>`` in the generated HTML."""
        content = html_path.read_text(encoding="utf-8")
        if "</head>" not in content:
            return
        css = (
            "<style>\n"
            "body { font-family: Arial, sans-serif; margin: 0; padding: 0; }\n"
            "#mynetwork { width: 100%; height: 100vh; }\n"
            "</style>\n"
        )
        content = content.replace("</head>", css + "</head>", 1)
        html_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_html(
        self,
        G: nx.DiGraph,
        *,
        node_size_attr: str = "citations",
        node_color_attr: str | None = None,
        title: str = "Citation Network",
    ) -> Path:
        """Generate an interactive HTML visualization using PyVis.

        Parameters
        ----------
        G:
            NetworkX directed graph.
        node_size_attr:
            Node attribute used to scale node sizes (default ``"citations"``).
        node_color_attr:
            Optional node attribute used for community-based coloring.
        title:
            Title rendered in the HTML page.

        Returns
        -------
        Path
            The path to the generated HTML file.
        """
        # Deferred (P7.7): module load stays stdlib-only
        from pyvis.network import Network

        node_sizes = self._compute_node_sizes(G, node_size_attr)
        node_colors = (
            self._compute_node_colors(G, node_color_attr) if node_color_attr else {}
        )

        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=True,
        )

        # Add nodes manually to inject computed size / color / tooltip
        for node, data in G.nodes(data=True):
            tooltip = self._build_tooltip(node, data)
            kwargs: dict[str, Any] = {
                "label": node,
                "title": tooltip,
                "value": node_sizes.get(node, 15),
            }
            if node_color_attr and node in node_colors:
                kwargs["color"] = node_colors[node]
            net.add_node(node, **kwargs)

        for src, dst in G.edges():
            net.add_edge(src, dst)

        # Physics
        physics_enabled_str = "true" if self.physics_enabled else "false"
        net.set_options(
            """{
          "nodes": {
            "shape": "dot",
            "scaling": { "min": 10, "max": 30 }
          },
          "edges": {
            "color": { "inherit": true },
            "smooth": false
          },
          "physics": {
            "enabled": %s,
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }"""
            % physics_enabled_str
        )

        net.heading = title

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(self.output_path))
        self._inject_custom_css(self.output_path)

        return self.output_path
