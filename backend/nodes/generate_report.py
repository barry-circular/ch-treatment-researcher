from datetime import datetime
from langchain_core.messages import AIMessage
from langchain_anthropic import ChatAnthropic
from ..classes import ResearchState



class GenerateNode:
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=0
        )
    def extract_markdown_content(self, content):
    # Strip out extra preamble or conversational text, retaining only Markdown.
        start_index_hash = content.find("#")
        start_index_bold = content.find("**")
        
        if start_index_hash != -1 and (start_index_bold == -1 or start_index_hash < start_index_bold):
            # '#' found and it comes before '**' (or '**' not found)
            return content[start_index_hash:].strip()
        elif start_index_bold != -1:
            # '**' found
            return content[start_index_bold:].strip()
        else:
            # Neither '#' nor '**' found, return the whole content stripped
            return content.strip()

    async def generate_report(self, state: ResearchState):
        report_title = f"Weekly Report on {state['treatment_name']}"
        report_date = datetime.now().strftime('%B %d, %Y')

        prompt = f"""
        You are an expert clinical researcher tasked with writing a fact-based report on recent developments for the treatment **{state['treatment_name']}**. Write the report in Markdown format, but **do not include a title**. Each section must be written in well-structured paragraphs, use lists or bullet points where appropriate.
        Ensure the report includes:
        - **Inline citations** as Markdown hyperlinks directly in the main sections (e.g., Treatment X is an innovative approach to Long Covid ([LinkedIn](https://linkedin.com))).
        - A **Citations Section** at the end that lists all URLs used.

        ### Report Structure:
        1. **Executive Summary**:
            - High-level overview of the treatment, its mechanism of action, its intended benefits, its safety profile, and its efficacy.
            - Make sure to include the general information necessary to understand the treatment well including any notable findings.

        2. **Mechanism of Action**:
            - Details on the mechanism of action of the treatment, how it works, what it does to the body, and how it is intended to benefit the patient.
            - Any relevant findings or evidence to support the mechanism of action.

        3. **Efficacy**:
            - Summary of current efficacy data, including any relevant medical studies or evidence to support the efficacy of the treatment.
            - Include details from the source website, tools, or new integrations.

        4. **Safety Profile**:
            - Summary of the safety profile of the treatment, including any relevant adverse effects or side effects.
            - Include details from the source website, tools, or new integrations.

        5. **Recent Developments**:
            - Recent developments in the treatment, including any relevant medical studies or evidence to support the efficacy of the treatment.
            - Include details from the source website, tools, or new integrations.

        6. **Citations**:
            - Ensure every source cited in the report is listed in the text as Markdown hyperlinks.
            - Also include a list of all URLs as Markdown hyperlinks in this section.

        ### Documents to Base the Report On:
        {state['documents']}
        """

        messages = [("system", "Your task is to generate a Markdown report."), ("human", prompt)]

        try:
            # Invoke the model
            response = await self.model.ainvoke(messages)

            # Extract the Markdown content
            markdown_content = self.extract_markdown_content(response.content)

            # Add the title and date to the response
            full_report = f"# {report_title}\n\n*{report_date}*\n\n{markdown_content}"
            return {"messages": [AIMessage(content=f"Report generated successfully!\n{full_report}")], "report": full_report}
        except Exception as e:
            error_message = f"Error generating report: {str(e)}"
            return {
                "messages": [AIMessage(content=error_message)],
                "report": f"# Error Generating Report\n\n*{report_date}*\n\n{error_message}"
            }


    async def run(self, state: ResearchState, websocket):
        if websocket:
            await websocket.send_text("⌛️ Generating report...")
        result = await self.generate_report(state)
        return result