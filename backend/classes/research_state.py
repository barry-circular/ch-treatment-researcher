from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from . import TavilySearchInput, DocumentCluster, ReportEvaluation
from typing import TypedDict, List, Annotated, Dict, Union

# Import directly from each specific module within format_classes
from .classes import TavilySearchInput, DocumentCluster, ReportEvaluation

# Define the research state
class ResearchState(TypedDict):
    treatment_name: str 
    source_url: str
    initial_documents: Dict[str, Dict[Union[str, int], Union[str, float]]]
    sub_questions: TavilySearchInput
    documents: Dict[str, Dict[Union[str, int], Union[str, float]]]
    document_clusters: List[DocumentCluster]
    chosen_cluster: int
    report: str
    eval: ReportEvaluation
    output_format: str
    messages: Annotated[list[AnyMessage], add_messages]

class InputState(TypedDict):
    treatment_name: str
    source_url: str


class OutputState(TypedDict):
    report: str