from pydantic import BaseModel, Field
from typing import List, Optional

# Add Tavily's arguments to enhance the web search tool's capabilities
class TavilyQuery(BaseModel):
    query: str = Field(description="web search query")
 

# Define the args_schema for the tavily_search tool using a multi-query approach, enabling more precise queries for Tavily.
class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="set of sub-queries that can be answered in isolation")


# Define the structure for clustering output
class DocumentCluster(BaseModel):
    treatment_name: str = Field(
        ...,
        description="The name or identifier of the company these documents belong to."
    )
    cluster: List[str] = Field(
        ...,
        description="A list of URLs relevant to the identified company."
    )

class DocumentClusters(BaseModel):
    clusters: List[DocumentCluster] = Field(default_factory=list, description="List of document clusters")

# Define the ReportEvaluation structure
class ReportEvaluation(BaseModel):
    grade: int = Field(
        ..., 
        description="Overall grade of the report on a scale from 1 to 3 (1 = needs improvement, 3 = complete and thorough)."
    )
    critical_gaps: Optional[List[str]] = Field(
        None, 
        description="List of critical gaps to address if the grade is 1."
    ) 
