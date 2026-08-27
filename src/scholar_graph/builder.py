import asyncio
import networkx as nx
from rich.progress import Progress
from scholar_search.http_client import AcademicHttpClient

from .models import GraphNode, GraphEdge
from .config import settings

class CitationGraphBuilder:
    def __init__(self, http_client: AcademicHttpClient):
        self.http_client = http_client
        
    async def fetch_work_data(self, doi: str) -> dict | None:
        """Fetch metadata for a single DOI from OpenAlex."""
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        try:
            response = await self.http_client.get(url=url)
            if response.status_code == 200:
                data = response.json()
                if data and data.get("id"):
                    return data
        except Exception:
            pass
        return None

    async def build_graph(self, dois: list[str], progress_callback=None) -> nx.DiGraph:
        """Build a directed citation graph from a list of DOIs."""
        G = nx.DiGraph()
        
        # 1. Fetch data for all DOIs
        works_data = []
        tasks = []
        for doi in dois:
            tasks.append(self.fetch_work_data(doi))
            
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if res:
                works_data.append(res)
            if progress_callback:
                progress_callback()
            
        # 2. Build mapping and add nodes
        wid_to_doi = {}
        for work in works_data:
            wid = work.get("id")
            doi_url = work.get("doi")
            if not wid or not doi_url:
                continue
                
            doi_clean = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")
            wid_to_doi[wid] = doi_clean
            
            title = work.get("title") or "Unknown Title"
            year = work.get("publication_year")
            citations = work.get("cited_by_count", 0)
            
            # Add node to networkx
            G.add_node(
                doi_clean, 
                title=title, 
                year=year, 
                citations=citations, 
                group=1,
                label=title[:30] + "..." if len(title) > 30 else title
            )
            
        # 3. Add edges (Citations)
        for work in works_data:
            source_wid = work.get("id")
            source_doi = wid_to_doi.get(source_wid)
            if not source_doi:
                continue
                
            refs = work.get("referenced_works", [])
            for ref_wid in refs:
                target_doi = wid_to_doi.get(ref_wid)
                if target_doi and source_doi != target_doi:
                    G.add_edge(source_doi, target_doi)
                    
        return G
