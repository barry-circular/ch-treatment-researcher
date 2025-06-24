from langchain_core.messages import AIMessage
from langchain_anthropic import ChatAnthropic

from ..classes import ResearchState,DocumentClusters



class ClusterNode:
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=0
        )

    async def cluster(self, state: ResearchState):
        treatment_name = state['treatment_name']
        source_url = state['source_url']
        initial_docs = state['initial_documents']
        documents = state.get('documents', {})
   
        # Extract compnay domain from URL
        target_domain = source_url.split("//")[-1].split("/")[0]

        # Collect all retrieved documents without duplicates
        unique_urls = []
        seen_urls = set()
        for url, doc, in documents.items():
            if url not in seen_urls:
                unique_urls.append({'url': url, 'content': doc.get('content', '')})
                seen_urls.add(url)

        # Pass in the first 25 URLs
        urls = unique_urls[:25]

        # LLM prompt to categorize documents accurately
        prompt = f"""
            We conducted a search for a treatment called '{treatment_name}', in relation to complex illness (eg Long Covid, ME/CFS, MCAS), but the results may include documents from other treatments with similar names or domains.
            It may also have found information related to use of the treatment outside of the context of the complex illness or even related to uses of the treatment on non humans.
            Your task is to accurately categorize these retrieved documents based on which specific treatment they pertain to, using the initial treatment information as "ground truth."

            ### Target Treatment Information
            - **Treatment Name**: '{treatment_name}'
            - **Primary Domain**: '{target_domain}'
            - **Initial Context (Ground Truth)**: Information below should act as a verification baseline. Use it to confirm that the document content aligns directly with {treatment_name} in relation to the complex illness (eg Long Covid, ME/CFS, MCAS).
            - **{initial_docs}**

            ### Retrieved Documents for Clustering
            Below are the retrieved documents, including URLs and brief content snippets:
            {[{'url': doc['url'], 'snippet': doc['content']} for doc in urls]}

            ### Clustering Instructions
            - **Treatment Name and Illness Priority**: Documents containing '{treatment_name}' should be prioritized for the main cluster for '{treatment_name}'.
            - **Include Relevant Third-Party Sources**: Documents from third-party domains (e.g., news sites, clinical reports) should also be included in the '{treatment_name}' cluster if they provide specific information about '{treatment_name}', reference the complex illness (eg Long Covid, ME/CFS, MCAS, or closely match the initial treatment context.
            - **Separate Similar But Ambigious**: Documents that clearly references '{treatment_name}') but that do no explicitly reference the complex illness (eg Long Covid, ME/CFS, MCAS) should be placed in separate clusters unless they explicitly reference the the treatment of human patients with complex illness.
            - **Handle Ambiguities Separately**: Documents that lack clear alignment with '{treatment_name}' should be placed in an "Ambiguous" cluster for further review.

            ### Example Output Format
            {{
                "clusters": [
                    {{
                        "treatment_name": "Name of Treatment A",
                        "cluster": [
                            "http://example.com/doc1",
                            "http://example.com/doc2"
                        ]
                    }},
                    {{
                        "treatment_name": "Name of Treatment B",
                        "cluster": [
                            "http://example.com/doc3"
                        ]
                    }},
                    {{
                        "treatment_name": "Ambiguous",
                        "cluster": [
                            "http://example.com/doc4"
                        ]
                    }}
                ]
            }}

            ### Key Points
            - **Focus on Relevant Content**: Documents that contain relevant references to '{treatment_name}' (even from third-party domains) should be clustered with '{treatment_name}' if they align well with the initial information and complext illness context provided.
            - **Identify Ambiguities**: Any documents without clear relevance to '{treatment_name}' and the complex illness (eg Long Covid, ME/CFS, MCAS) should be placed in the "Ambiguous" cluster for manual review.
        """

        # LLM call with structured output using DocumentClusters
        messages = ["system","Your job is to generate clusters for the treatment: '{treatment_name}'.\n",
                ("human",f"{prompt}")]
        
        msg = ""
        try:
            # Use the model's structured output with DocumentClusters format
            response = await self.model.with_structured_output(DocumentClusters).ainvoke(messages)
            clusters = response.clusters  # Access the structured clusters directly
      
        except Exception as e:
            msg = f"Error: {str(e)}\n"
            clusters = []


        # Summarize the results
        if not clusters:
            msg += "No valid clusters generated. Please check the document formats.\n"
        else:
            msg += "Clusters generated successfully:\n"
            urls = set()
            for  idx, cluster in enumerate(clusters, start=1):
                msg += f"   📂 Treatment {idx}: {cluster.treatment_name}\n"
                for url in cluster.cluster:
                    domain = url.split("://")[-1].split("/")[0]
                    if domain not in urls:
                        urls.add(domain)
                        msg += f"       📄 {domain}\n"
        
        return {"messages": [AIMessage(content=msg)], "document_clusters": clusters}
    
    # Define the function to choose the correct cluster as a conditional edge
    async def choose_cluster(self, state: ResearchState):
        source_url = state['source_url']
        clusters = state['document_clusters']

        # Attempt to automatically choose the correct cluster
        for index,cluster in enumerate(clusters):
            # Check if any URL in the cluster starts with the company URL
            if any(url.startswith(source_url) for url in cluster.cluster):
                # state['chosen_cluster'] = index
                msg = f"Automatically selected cluster for '{source_url}' as {cluster.treatment_name}."
                return {"messages": [AIMessage(content=msg)], "chosen_cluster": index}

        # If no automatic match, indicate that manual selection is needed
        msg = "No automatic cluster match found. Please select the correct cluster manually."
        return {"messages": [AIMessage(content=msg)], "document_clusters": clusters, "chosen_cluster": None}

    async def run(self, state: ResearchState, websocket):
        if websocket:
            await websocket.send_text("🔄 Beginning clustering process...")

        cluster_result = await self.cluster(state)
        state['document_clusters'] = cluster_result['document_clusters'] 
        choose_cluster_result = await self.choose_cluster(state)
        result = {'chosen_cluster': choose_cluster_result['chosen_cluster']}
        result.update(cluster_result)
        return result