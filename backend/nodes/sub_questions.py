from langchain_core.messages import AIMessage
from langchain_anthropic import ChatAnthropic
from ..classes import ResearchState, TavilySearchInput

class SubQuestionsNode:
    def __init__(self) -> None:
        self.model = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=0
        )
     
    # Function to generate sub-questions based on initial search data
    async def generate_sub_questions(self, state: ResearchState):
        try:
            msg = "🤔 Generating sub-questions based on the initial search results...\n"
            
            if 'sub_questions_data' not in state:
                state['sub_questions_data'] = []
                
            # Prompt to generate detailed sub-questions
            prompt = f"""
            You are an expert clinical researcher focusing on treatment analysis, in relation to complex illness (eg Long Covid, ME/CFS, MCAS), in order to generate a report.
            Your task is to generate 5 specific sub-questions that will provide a thorough understanding of the treatment: '{state['treatment_name']}'.
            
            ### Key Areas to Explore:
            - **Description**: What is the intended purpose of the treatment in relation to the complex illness (eg Long Covid, ME/CFS, MCAS)?
            - **Mechanism of action**: How it works, what affect does it have on the body, what are its intended benefits?
            - **Efficacy**: What is the efficacy of the treatment?
            - **Safety**: What is the safety of the treatment?
            - **Adverse effects**: What are the adverse effects of the treatment?

            Use the initial information provided from the source website below to keep questions directly relevant to **{state['treatment_name']}**.

            Source URL: {state['source_url']}
            Initial Treatment Information:
            {state["initial_documents"]}
            
            Ensure questions are clear, specific, and well-aligned with the treatment in the context of treating patients with the complex illness mentioned above.
            """
            
            # Use LLM to generate sub-questions
            messages = ["system","Your task is to generate sub-questions based on the initial search results.",
                ("human",f"{prompt}")]

            sub_questions = await self.model.with_structured_output(TavilySearchInput).ainvoke(messages)
            
        except Exception as e:
            msg = f"An error occurred during sub-question generation: {str(e)}"
            return {"messages": [AIMessage(content=msg)], "sub_questions": None, "initial_documents": state['initial_documents']}
            
        
        return {"messages": [AIMessage(content=msg)], "sub_questions": sub_questions, "initial_documents": state['initial_documents']}
            
    async def run(self, state: ResearchState):
        result = await self.generate_sub_questions(state)
        return result