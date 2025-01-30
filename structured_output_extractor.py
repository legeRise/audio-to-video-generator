from typing import Type, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import constants  # Assuming constants.py holds LLM provider configurations
from langchain_groq import ChatGroq


# Define the State structure (similar to previous definition)
class State(TypedDict):
    messages: list
    output: Optional[BaseModel]


# Generic Pydantic model-based structured output extractor
class StructuredOutputExtractor:
    def __init__(self, response_schema: Type[BaseModel]):
        """
        Initializes the extractor for any given structured output model.
        
        :param response_schema: Pydantic model class used for structured output extraction
        """
        self.response_schema = response_schema

        # Initialize language model (provider and API keys come from constants.py)
        # self.llm = ChatGroq(model="llama-3.3-70b-versatile")  # token limit 100k tokens
        self.llm = ChatGroq(model="deepseek-r1-distill-llama-70b")  # currently no limit per day
        
        # Bind the model with structured output capability
        self.structured_llm = self.llm.with_structured_output(response_schema)
        
        # Build the graph for structured output
        self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph computational graph for structured extraction.
        """
        graph_builder = StateGraph(State)

        # Add nodes and edges for structured output
        graph_builder.add_node("extract", self._extract_structured_info)
        graph_builder.add_edge(START, "extract")
        graph_builder.add_edge("extract", END)

        self.graph = graph_builder.compile()

    def _extract_structured_info(self, state: dict):
        """
        Extract structured information using the specified response model.
        
        :param state: Current graph state
        :return: Updated state with structured output
        """
        query = state['messages'][-1].content
        print(f"Processing query: {query}")
        try:
            # Extract details using the structured model
            output = self.structured_llm.invoke(query)
            # Return the structured response
            return {"output": output}
        except Exception as e:
            print(f"Error during extraction: {e}")
            return {"output": None}

    def extract(self, query: str) -> Optional[BaseModel]:
        """
        Public method to extract structured information.
        
        :param query: Input query for structured output extraction
        :return: Structured model object or None
        """
        from langchain_core.messages import SystemMessage

        result = self.graph.invoke({
            "messages": [SystemMessage(content=query)]
        })
        # Return the structured model response, if available
        result = result.get('output')
        return result


if __name__ == '__main__':
        
        # Example Pydantic model (e.g., Movie)
        class Movie(BaseModel):
            title: str
            year: int
            genre: str
            rating: Optional[float] = None
            actors: list[str] = []


        # Example usage with a generic structured extractor
        extractor = StructuredOutputExtractor(response_schema=Movie)

        query = "Tell me about the movie Inception. Provide details about its title, year, genre, rating, and main actors."

        result = extractor.extract(query)
        print(type(result))
        if result:
            print(result)