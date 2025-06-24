from langchain_core.messages import AIMessage
from tavily import AsyncTavilyClient
import os

from ..classes import ResearchState


class InitialGroundingNode:
    def __init__(self) -> None:
        self.tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Use Tavily Extract to get base content from provided company URL
    async def initial_search(self, state: ResearchState):
        msg = f"🔎 Initiating initial grounding for treatment '{state['treatment_name']}'...\n"

        urls = []
        urls.append(state['source_url'])
        state['initial_documents'] = {}
        
        try:
            search_results = await self.tavily_client.extract(urls=urls)
            for item in search_results["results"]:
                url = item['url']
                raw_content = item["raw_content"]
                state['initial_documents'][url] = {'url': url, 'raw_content': raw_content}
                # msg += f"Extracted raw content for URL: {url}\n"
                
        except Exception as e:
            print(f"Error occurred during Tavily Extract request:{e}")
        
        return {"messages": [AIMessage(content=msg)], "initial_documents": state['initial_documents']}
    
    async def run(self, state: ResearchState):
        result = await self.initial_search(state)
        return result

    