from langchain_core.messages import AIMessage
from tavily import AsyncTavilyClient
import os
import asyncio
from datetime import datetime
from typing import List


from ..classes import ResearchState, TavilyQuery

class ResearcherNode():
    def __init__(self):
        self.tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


    async def tavily_search(self, sub_queries: List[TavilyQuery]):
        """Perform searches for each sub-query using the Tavily search tool concurrently."""  
        # Define a coroutine function to perform a single search with error handling
        async def perform_search(itm):
            try:
                # Add date to the query as we need the most recent results
                query_with_date = f"{itm.query} {datetime.now().strftime('%m-%Y')}"
                # Attempt to perform the search, hardcoding days to 7 (days will be used only when topic is news)
                response = await self.tavily_client.search(query=query_with_date, topic="general", max_results=7)
                return response['results']
            except Exception as e:
                # Handle any exceptions, log them, and return an empty list
                print(f"Error occurred during search for query '{itm.query}': {str(e)}")
                return []
        
        # Run all the search tasks in parallel
        search_tasks = [perform_search(itm) for itm in sub_queries]
        search_responses = await asyncio.gather(*search_tasks)
        
        # Combine the results from all the responses
        search_results = []
        for response in search_responses:
            search_results.extend(response)
        
        return search_results


    async def research(self, state: ResearchState):
        """
        Conducts a Tavily Search and stores all documents in a unified 'documents' attribute.
        """
        msg = "🚀 Conducting Tavily Search for the specified treatment...\n"
        state['documents'] = {}  # Initialize documents if not already present

        research_node = ResearcherNode()
        # Perform the search and gather results
        response = await research_node.tavily_search(state['sub_questions'].sub_queries)

        # Process each set of search results and add to documents
        for doc in response:
            url = doc.get('url')
            if url and url not in state['documents']:  # Avoid duplicates
                state['documents'][url] = doc

        return {"messages": [AIMessage(content=msg)], "documents": state['documents']}
    
    async def run(self, state: ResearchState):
        result = await self.research(state)
        return result