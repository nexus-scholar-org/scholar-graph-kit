import networkx as nx
from pathlib import Path
from pyvis.network import Network

class GraphVisualizer:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        
    def generate_html(self, G: nx.DiGraph):
        """Generate an interactive HTML visualization using PyVis."""
        # Scale nodes based on citations and prepare tooltips
        for node, data in G.nodes(data=True):
            citations = data.get("citations", 0)
            data["value"] = citations + 5  # Base size so nodes aren't too small
            data["title"] = f"DOI: {node}\nTitle: {data.get('title')}\nYear: {data.get('year')}\nCitations: {citations}"
            
        # Create pyvis network
        net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=True)
        net.from_nx(G)
        
        # Configure physics for a pleasant academic layout
        net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "scaling": {
              "min": 10,
              "max": 30
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": false
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        
        # Save output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(self.output_path))
