from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from .builder import CitationGraphBuilder
from .visualizer import GraphVisualizer

# Force UTF-8 on Windows to prevent Rich console UnicodeEncodeError on OEM code pages
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = typer.Typer(
    help="Scholar Graph Kit: Build and visualize citation graphs from Open Access DOIs.",
    no_args_is_help=True,
)
console = Console()


@app.command("build")
def build(
    dois: list[str] | None = typer.Option(
        None,
        "--doi",
        "-d",
        help="Specific DOI to map (can be specified multiple times)",
    ),
    input_file: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="JSON file containing results from scholar-search-kit",
    ),
    output_file: Path = typer.Option(
        Path("graph.html"),
        "--output",
        "-o",
        help="Path to save the output HTML visualization",
    ),
    json_output: Path | None = typer.Option(
        None,
        "--json-output",
        "-j",
        help="Path to save graph topology and PageRank JSON",
    ),
    export_format: str = typer.Option(
        "html+json",
        "--format",
        "-f",
        help="Export formats: html, json, gexf, graphml, all",
    ),
    mode: str = typer.Option(
        "citation",
        "--mode",
        "-m",
        help="Graph mode: citation, cocitation, coupling, hybrid",
    ),
):
    """Build a citation graph from DOIs and generate an interactive HTML map."""
    doi_list = list(dois) if dois else []

    # Parse input file if provided
    if input_file and input_file.exists():
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    doi = None
                    if "external_ids" in item and item["external_ids"].get("doi"):
                        doi = item["external_ids"]["doi"]
                    elif "doi" in item:
                        doi = item["doi"]

                    if doi and doi not in doi_list:
                        doi_list.append(doi)
        except Exception as e:
            console.print(f"[bold red]Error parsing input file:[/bold red] {e}")
            raise typer.Exit(1)

    if not doi_list:
        console.print("[bold yellow]No DOIs provided to build graph.[/bold yellow]")
        raise typer.Exit(0)

    console.print(
        f"[bold blue]Starting graph build for {len(doi_list)} DOIs...[/bold blue]"
    )

    async def run_build():
        from scholar_search.http_client import AcademicHttpClient

        http_client = AcademicHttpClient(name="openalex-graph", rate_limit=10)
        builder = CitationGraphBuilder(http_client)

        with console.status("[cyan]Fetching citations from OpenAlex...") as status:

            def update_progress():
                pass  # Simple callback

            G = await builder.build_graph(doi_list, progress_callback=update_progress)

        return G, builder

    G, builder = asyncio.run(run_build())

    console.print(
        f"[bold green]Citation graph built![/bold green] (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})"
    )

    # Apply mode-specific scientometric transformation
    if mode == "cocitation":
        from .scientometrics import ScientometricEngine

        engine = ScientometricEngine(G)
        G = engine.build_cocitation_network(min_jaccard=0.15)
        console.print(
            f"[green]Co-citation network: {len(G.nodes)} nodes, {len(G.edges)} edges[/]"
        )
    elif mode == "coupling":
        from .scientometrics import ScientometricEngine

        engine = ScientometricEngine(G)
        G = engine.build_bibliographic_coupling(min_jaccard=0.15)
        console.print(
            f"[green]Coupling network: {len(G.nodes)} nodes, {len(G.edges)} edges[/]"
        )
    elif mode == "hybrid":
        from .scientometrics import ScientometricEngine

        engine = ScientometricEngine(G)
        G = engine.build_hybrid_network(alpha=0.5)
        console.print(
            f"[green]Hybrid network: {len(G.nodes)} nodes, {len(G.edges)} edges[/]"
        )
    elif mode == "citation":
        pass  # Existing behavior, no transformation needed
    else:
        raise typer.BadParameter(
            f"Unknown mode: {mode}. Use citation, cocitation, coupling, or hybrid."
        )

    console.print(f"[cyan]Rendering PyVis visualization to {output_file}...[/cyan]")
    vis = GraphVisualizer(output_file)
    vis.generate_html(G)

    console.print(f"[bold green]Saved visualization to {output_file}[/bold green]")

    if json_output:
        builder.export_json(G, json_output)
        console.print(
            f"[bold green]Saved graph JSON & PageRank to {json_output}[/bold green]"
        )
    else:
        # Default adjacent json if not specified
        default_json = output_file.with_suffix(".json")
        builder.export_json(G, default_json)

    formats = [f.strip() for f in export_format.split("+")]

    if "gexf" in formats or "all" in formats:
        gexf_path = output_file.with_suffix(".gexf")
        CitationGraphBuilder.export_gexf(G, gexf_path)
        console.print(f"[bold green]GEXF exported to {gexf_path}[/bold green]")

    if "graphml" in formats or "all" in formats:
        graphml_path = output_file.with_suffix(".graphml")
        CitationGraphBuilder.export_graphml(G, graphml_path)
        console.print(f"[bold green]GraphML exported to {graphml_path}[/bold green]")


@app.command("pagerank")
def pagerank(
    graph_file: Path = typer.Argument(
        ..., help="Path to graph.json exported by build command"
    ),
):
    """Display PageRank scores computed from a citation graph."""
    if not graph_file.exists():
        console.print(f"[bold red]Error:[/bold red] Graph file {graph_file} not found.")
        raise typer.Exit(1)

    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pr = data.get("pagerank", {})
    if not pr:
        console.print("[yellow]No PageRank scores found in graph JSON.[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"PageRank Scores ({graph_file.name})")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("DOI / Identifier", style="white")
    table.add_column("Normalized PageRank", style="magenta")

    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    for idx, (node_id, score) in enumerate(sorted_pr, start=1):
        table.add_row(str(idx), node_id, f"{score:.4f}")

    console.print(table)


@app.command("cluster")
def cluster(
    graph_file: Path = typer.Argument(..., help="Path to graph JSON file"),
    resolution: float = typer.Option(
        1.0, "--resolution", "-r", help="Louvain resolution parameter"
    ),
    seed: int = typer.Option(
        42, "--seed", "-s", help="Random seed for reproducibility"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Detect communities using Louvain algorithm."""
    import networkx as nx

    from .scientometrics import ScientometricEngine

    if not graph_file.exists():
        console.print(f"[bold red]Error:[/bold red] Graph file {graph_file} not found.")
        raise typer.Exit(1)

    with open(graph_file, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.node_link_graph(data)

    engine = ScientometricEngine(G)

    # Detect communities
    communities = engine.detect_communities_louvain(seed=seed, resolution=resolution)
    modularity = engine.compute_modularity(communities)
    engine.enrich_graph_with_communities(communities)

    # Build result
    community_sizes: dict[int, int] = {}
    for node, comm_id in communities.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

    result = {
        "num_communities": len(set(communities.values())),
        "modularity": modularity,
        "community_sizes": community_sizes,
        "node_community_map": communities,
    }

    result_str = json.dumps(result, indent=2)

    if output:
        output.write_text(result_str, encoding="utf-8")
        console.print(f"[green]Results written to {output}[/]")
    else:
        console.print(result_str)

    console.print(f"\n[bold]Communities:[/] {result['num_communities']}")
    console.print(f"[bold]Modularity:[/] {modularity:.4f}")


@app.command("analyze")
def analyze(
    graph_file: Path = typer.Argument(..., help="Path to graph JSON file"),
    metric: str = typer.Option(
        "hits", "--metric", "-m", help="Metric: hits, betweenness"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Analyze citation graph with scientometric metrics."""
    import networkx as nx

    from .scientometrics import ScientometricEngine

    if not graph_file.exists():
        console.print(f"[bold red]Error:[/bold red] Graph file {graph_file} not found.")
        raise typer.Exit(1)

    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.node_link_graph(data)
    engine = ScientometricEngine(G)

    if metric == "hits":
        hubs, authorities = engine.compute_hits()
        classification = engine.classify_nodes_by_hits(hubs, authorities)
        result = {
            "hubs": hubs,
            "authorities": authorities,
            "classification": classification,
        }
    elif metric == "betweenness":
        centrality = engine.compute_betweenness_centrality()
        result = {"centrality": centrality}
    else:
        console.print(
            f"[bold red]Error:[/bold red] Unknown metric: {metric}. Use 'hits' or 'betweenness'."
        )
        raise typer.BadParameter(
            f"Unknown metric: {metric}. Use 'hits' or 'betweenness'."
        )

    output_str = json.dumps(result, indent=2)

    if output:
        output.write_text(output_str)
        console.print(f"[bold green]Results written to {output}[/bold green]")
    else:
        typer.echo(output_str)


@app.command("visualize")
def visualize_cmd(
    graph_file: Path = typer.Argument(
        ..., help="Path to graph JSON file (node-link format)"
    ),
    output: Path = typer.Option(
        Path("graph.html"),
        "--output",
        "-o",
        help="Path to save the output HTML visualization",
    ),
    title: str = typer.Option(
        "Citation Network",
        "--title",
        "-t",
        help="Title for the visualization",
    ),
    physics: bool = typer.Option(
        True,
        "--physics/--no-physics",
        help="Enable or disable physics simulation",
    ),
    node_size: str = typer.Option(
        "citations",
        "--node-size",
        help="Node attribute for sizing (default: citations)",
    ),
    node_color: str | None = typer.Option(
        None,
        "--node-color",
        help="Node attribute for community coloring (e.g. community)",
    ),
):
    """Load a graph JSON file and render an interactive HTML visualization."""
    # Deferred (P7.7): heavy imports inside function body
    import json

    import networkx as nx

    if not graph_file.exists():
        console.print(f"[bold red]Error:[/bold red] Graph file {graph_file} not found.")
        raise typer.Exit(1)

    data = json.loads(graph_file.read_text(encoding="utf-8"))
    G = nx.node_link_graph(data)

    vis = GraphVisualizer(
        output,
        physics_enabled=physics,
    )
    vis.generate_html(
        G,
        node_size_attr=node_size,
        node_color_attr=node_color,
        title=title,
    )

    console.print(
        f"[bold green]Saved visualization to {output}[/bold green] "
        f"(Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})"
    )


if __name__ == "__main__":
    app()
