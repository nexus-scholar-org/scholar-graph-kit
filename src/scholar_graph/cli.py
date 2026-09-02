from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .builder import CitationGraphBuilder
from .config import settings
from .visualizer import GraphVisualizer

app = typer.Typer(
    help="Scholar Graph Kit: Build and visualize citation graphs from Open Access DOIs.",
    no_args_is_help=True,
)
console = Console()


@app.command("build")
def build(
    dois: Optional[list[str]] = typer.Option(None, "--doi", "-d", help="Specific DOI to map (can be specified multiple times)"),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="JSON file containing results from scholar-search-kit"),
    output_file: Path = typer.Option(Path("graph.html"), "--output", "-o", help="Path to save the output HTML visualization"),
    json_output: Optional[Path] = typer.Option(None, "--json-output", "-j", help="Path to save graph topology and PageRank JSON"),
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
        
    console.print(f"[bold blue]Starting graph build for {len(doi_list)} DOIs...[/bold blue]")
    
    async def run_build():
        from scholar_search.http_client import AcademicHttpClient
        http_client = AcademicHttpClient(name="openalex-graph", rate_limit=10)
        builder = CitationGraphBuilder(http_client)
        
        with console.status("[cyan]Fetching citations from OpenAlex...") as status:
            def update_progress():
                pass # Simple callback
            G = await builder.build_graph(doi_list, progress_callback=update_progress)
            
        return G, builder
        
    G, builder = asyncio.run(run_build())
    
    console.print(f"[bold green]Graph built successfully![/bold green] (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
    
    console.print(f"[cyan]Rendering PyVis visualization to {output_file}...[/cyan]")
    vis = GraphVisualizer(output_file)
    vis.generate_html(G)
    
    console.print(f"[bold green]Saved visualization to {output_file}[/bold green]")

    if json_output:
        builder.export_json(G, json_output)
        console.print(f"[bold green]Saved graph JSON & PageRank to {json_output}[/bold green]")
    else:
        # Default adjacent json if not specified
        default_json = output_file.with_suffix(".json")
        builder.export_json(G, default_json)


@app.command("pagerank")
def pagerank(
    graph_file: Path = typer.Argument(..., help="Path to graph.json exported by build command"),
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


if __name__ == "__main__":
    app()
