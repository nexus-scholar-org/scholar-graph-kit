import typer
import json
import asyncio
from pathlib import Path
from rich.console import Console

from .builder import CitationGraphBuilder
from .visualizer import GraphVisualizer
from .config import settings

app = typer.Typer(help="Scholar Graph Kit: Build and visualize citation graphs from Open Access DOIs.")
console = Console()

@app.command("build")
def build(
    dois: list[str] = typer.Option(None, "--doi", "-d", help="Specific DOI to map (can be specified multiple times)"),
    input_file: Path = typer.Option(None, "--input", "-i", help="JSON file containing results from scholar-search-kit"),
    output_file: Path = typer.Option(Path("graph.html"), "--output", "-o", help="Path to save the output HTML visualization")
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
            
        return G
        
    G = asyncio.run(run_build())
    
    console.print(f"[bold green]Graph built successfully![/bold green] (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
    
    console.print(f"[cyan]Rendering PyVis visualization to {output_file}...[/cyan]")
    vis = GraphVisualizer(output_file)
    vis.generate_html(G)
    
    console.print(f"[bold green]Saved visualization to {output_file}[/bold green]")
    console.print("You can open this file in any web browser.")

if __name__ == "__main__":
    app()
