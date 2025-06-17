from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import Annotated, Union
from langgraph.graph.message import add_messages

from message_helper import create_message_with_history, extract_last_message

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
        message_type: Classification result ("emotional" or "logical")
    """
    messages: Annotated[list, add_messages]  # Auto-managed by LangGraph
    message_type: Union[str, None]  # Can be None initially

# =============================================================================
# Define Agents
# =============================================================================

def emotional_agent(state: State) -> dict:
    """
    Emotional Agent: Provides empathetic, supportive responses.

    Specializes in:
    - Emotional support and validation
    - Therapeutic conversation techniques
    - Empathy and active listening
    - Mental health guidance (non-professional)

    Args:
        state: Current conversation state

    Returns:
        dict: State update with assistant's emotional response
    """

    # Emotional agent system prompt
    system_prompt = """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""

    # message_content = extract_last_message(state)
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": message_content}
    # ]

    messages = create_message_with_history(state, system_prompt)

    try:
        # Generate empathetic response
        response = lama3_2_llm.invoke(messages)
        print("💝 Emotional Agent responding...")

        return {"messages": [{"role": "assistant", "content": response.content}]}

    except Exception as e:
        error_msg = "I'm sorry, I'm having trouble responding right now. Please know that your feelings are valid and important. 💙"
        print(f"❌ Emotional agent error: {e}")
        return {"messages": [{"role": "assistant", "content": error_msg}]}


def logical_agent(state: State) -> dict:
    """
    Logical Agent: Provides factual, analytical responses.

    Specializes in:
    - Factual information and explanations
    - Logical reasoning and analysis
    - Step-by-step problem-solving
    - Technical and educational content

    Args:
        state: Current conversation state

    Returns:
        dict: State update with assistant's logical response
    """

    # Logical agent system prompt
    system_prompt = """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""

    # message_content = extract_last_message(state)
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": message_content}
    # ]
    #
    messages = create_message_with_history(state, system_prompt);

    try:
        # Generate logical response
        response = phi3_llm.invoke(messages)
        print("🧠 Logical Agent responding...")

        return {"messages": [{"role": "assistant", "content": response.content}]}

    except Exception as e:
        error_msg = "I apologize, but I'm experiencing technical difficulties. Please try rephrasing your question or check back in a moment."
        print(f"❌ Logical agent error: {e}")
        return {"messages": [{"role": "assistant", "content": error_msg}]}

