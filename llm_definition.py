from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import Annotated, Union
from langgraph.graph.message import add_messages

# =============================================================================
# CONFIGURATION
# =============================================================================

# Initialize Ollama LLM
lama3_2_llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7  # Balanced creativity/consistency
)

# Initialize Phi3 LLM
phi3_llm = ChatOllama(
    model="phi3:mini",
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7  # Balanced creativity/consistency
)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class State(TypedDict):
    """
    State structure for the conversation flow.

    Fields:
        messages: List of conversation messages (automatically merged by LangGraph)
        message_type: Classification result ("weather", "logic_gate", "emotional" or "logical")
    """
    messages: Annotated[list, add_messages]  # Auto-managed by LangGraph
    message_type: Union[str, None]  # Can be None initially