from langchain_core.messages import AIMessage
from tavily import AsyncTavilyClient
import os
from ..classes import ResearchState

class EnrichDocsNode:
    """
    Curates documents based on the selected cluster stored in `chosen_cluster`,
    then enriches the content with Tavily Extract for more detailed information.
    """
    def __init__(self):
        self.tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    async def curate(self, state: ResearchState):
        chosen_cluster_index = state['chosen_cluster']
        clusters = state['document_clusters']
        chosen_cluster = clusters[chosen_cluster_index]
        msg = f"🚀 Enriching documents for selected cluster '{chosen_cluster.treatment_name}'...\n"

        # Filter `documents` to include only those in the chosen cluster
        selected_docs = {url: state['documents'][url] for url in chosen_cluster.cluster if url in state['documents']}

        # Limit to first 15 URLs 
        urls_to_extract = list(selected_docs.keys())[:15]
        
        # Enrich the content using Tavily Extract
        try:
            extracted_content = await self.tavily_client.extract(urls=urls_to_extract)
            enriched_docs = {}
            
            # Update `documents` with enriched content from Tavily Extract
            for item in extracted_content["results"]:
                url = item['url']
                if url in selected_docs:
                    enriched_docs[url] = {
                        **selected_docs[url],  # Existing doc data
                        "raw_content": item.get("raw_content", ""),
                        "extracted_details": item.get("details", {}),
                    }
            
            state['documents'] = enriched_docs  # Update documents with enriched data

        except Exception as e:
            msg += f"Error occurred during Tavily Extract: {str(e)}\n"
            msg += f"Extracted URLs: {urls_to_extract}\n"  # Log the urls_to_extract if error

        return {"messages": [AIMessage(content=msg)], "documents": state['documents']}
    
    async def run(self, state: ResearchState):
        result = await self.curate(state)
        return result