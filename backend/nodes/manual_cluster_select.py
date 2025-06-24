# In your node file
from langchain_core.messages import AIMessage
from langgraph.errors import NodeInterrupt
from ..classes import ResearchState

class ManualSelectionNode:
    async def manual_cluster_selection(self, state: ResearchState, websocket):
        clusters = state['document_clusters']
        msg = "Multiple clusters were identified. Please review the options and select the correct cluster for the target treatment.\n\n"
        msg += "Enter '0' if none of these clusters match the target treatment.\n"

        if websocket:
            # Send cluster options to the frontend via WebSocket
            await websocket.send_text(msg)

            # Wait for user selection from WebSocket
            while True:
                try:
                    selection_text = await websocket.receive_text()
                    selected_cluster_index = int(selection_text) - 1

                    if selected_cluster_index == -1:
                        msg = "No suitable cluster found. Trying to cluster again.\n"
                        await websocket.send_text(msg)
                        return {"messages": [AIMessage(content=msg, is_manual_selection=True)], "chosen_cluster": selected_cluster_index}
                    elif 0 <= selected_cluster_index < len(clusters):
                        chosen_cluster = clusters[selected_cluster_index]
                        msg = f"You selected cluster '{chosen_cluster.treatment_name}' as the correct cluster."
                        await websocket.send_text(msg)
                        return {"messages": [AIMessage(content=msg, is_manual_selection=True)], "chosen_cluster": selected_cluster_index}
                    else:
                        await websocket.send_text("Invalid choice. Please enter a number corresponding to the listed clusters or '0' to re-cluster.")
                except ValueError:
                    await websocket.send_text("Invalid input. Please enter a valid number.")
        else:
            # Handle selection for studio, attempt to cluster again for now
            
            msg = "Manual selection needed, trying to cluster again.\n"
            return {"messages": [AIMessage(content=msg, is_manual_selection=True)], "chosen_cluster": -1}
            # selected_cluster_index = state.get('chosen_cluster', -1)  # Default to -1 if not set
            # if selected_cluster_index == -1:
            #     raise NodeInterrupt(
            #         "Please input the chosen cluster index for manual selection in LangGraph Studio. "
            #         "Set the chosen cluster index in the state attribute 'chosen_cluster'."
            #     )
            #     msg = "No suitable cluster found. Trying to cluster again.\n"
            #     return {"messages": [AIMessage(content=msg, is_manual_selection=True)], "chosen_cluster": selected_cluster_index}
            # elif 0 <= selected_cluster_index < len(clusters):
            #     chosen_cluster = clusters[selected_cluster_index]
            #     msg = f"You selected cluster '{chosen_cluster.company_name}' as the correct cluster."
            #     return {"messages": [AIMessage(content=msg, is_manual_selection=True)], "chosen_cluster": selected_cluster_index}
            # else:
            #     msg = "Invalid cluster selection in state. Please provide a valid cluster index."
            #     return {"messages": [AIMessage(content=msg)], "chosen_cluster": None}
    async def run(self, state: ResearchState, websocket=None):
        return await self.manual_cluster_selection(state, websocket)
