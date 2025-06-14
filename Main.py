"""
Simplified Dual-Agent Chatbot using LangGraph and Ollama

This chatbot routes user messages to one of two specialized agents:
1. Emotional Agent: Provides empathy, emotional support, and therapeutic responses
2. Logical Agent: Provides factual information, analysis, and logical reasoning

Architecture:
    START → Classifier → Router → [Emotional Agent | Logical Agent] → END

Requirements:
    - Ollama running locally with llama3.2:1b model
    - pip install langchain-ollama langgraph python-dotenv

Author: Mazhar Hassan
Date: 2025
"""

from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from agent_helpers import State, emotional_agent, logical_agent, llm
from message_helper import print_history

# Load environment variables
load_dotenv()

# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class MessageClassifier(BaseModel):
    """
    Pydantic model for automatic message classification with validation.

    This ensures the LLM returns exactly one of the specified message types,
    with automatic validation and error handling.
    """
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def classify_message(state: State) -> dict:
    """
    Classify the user's message as either 'emotional' or 'logical' using structured output.

    Args:
        state: Current conversation state containing messages

    Returns:
        dict: Updated state with message_type classification

    Note:
        Uses Pydantic MessageClassifier for automatic validation and structured output.
        Falls back to manual parsing if structured output fails (for local models).
    """
    # Get the last message from user
    last_message = state["messages"][-1]

    # Extract content from message (handle both dict and AIMessage objects)
    if hasattr(last_message, 'content'):  # AIMessage object
        message_content = last_message.content
    elif isinstance(last_message, dict):  # Dict format
        message_content = last_message.get("content", "")
    else:
        message_content = str(last_message)

    # System prompt for classification
    system_prompt = """You are a message classifier. Analyze the user's message and classify it as either:

- "emotional": if the message asks for emotional support, therapy, deals with feelings, personal problems, relationships, stress, anxiety, sadness, happiness, mental health, or seeks empathy and understanding

- "logical": if the message asks for facts, information, explanations, how-to guides, technical questions, analysis, calculations, or seeks objective knowledge

Respond with the classification in the exact format requested."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Classify this message: {message_content}"}
    ]

    try:
        # First, try using structured output with Pydantic validation
        classifier_llm = llm.with_structured_output(MessageClassifier)
        result = classifier_llm.invoke(messages)

        print(f"🔍 Classification (Structured): '{message_content[:50]}...' → {result.message_type}")
        return {"message_type": result.message_type}

    except Exception as structured_error:
        print(f"⚠️ Structured output failed: {structured_error}")
        print("🔄 Falling back to manual parsing...")

        try:
            # Fallback: Manual parsing for local models that don't support structured output
            classification_prompt = f"""
            Classify this user message as EXACTLY one word: either "emotional" or "logical"
            
            - "emotional": emotional support, therapy, feelings, personal problems, relationships, stress, anxiety, mental health
            - "logical": facts, information, explanations, technical questions, analysis, calculations
            
            User message: "{message_content}"
            
            Classification (one word only):
            """

            result = llm.invoke([{"role": "user", "content": classification_prompt}])
            classification = result.content.strip().lower()

            # Determine message type with fallback to logical
            if "emotional" in classification:
                message_type = "emotional"
            else:
                message_type = "logical"  # Default fallback

            print(f"🔍 Classification (Manual): '{message_content[:50]}...' → {message_type}")
            return {"message_type": message_type}

        except Exception as manual_error:
            print(f"❌ Both classification methods failed: {manual_error}")
            print("🔄 Defaulting to logical agent...")
            return {"message_type": "logical"}


def route_to_agent(state: State) -> dict:
    """
    Route the conversation to the appropriate agent based on classification.

    Args:
        state: Current conversation state with message_type

    Returns:
        dict: Routing decision for LangGraph
    """
    message_type = state.get("message_type", "logical")

    if message_type == "emotional":
        next_node = "emotional_agent"
    else:
        next_node = "logical_agent"

    print(f"🔀 Routing to: {next_node}")
    return {"next": next_node}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_chatbot_graph() -> StateGraph:
    """
    Construct the conversation flow graph.

    Flow:
        START → classify_message → route_to_agent → [emotional_agent | logical_agent] → END

    Returns:
        StateGraph: Compiled conversation graph
    """
    # Initialize graph builder
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("classify_message", classify_message)
    graph_builder.add_node("route_to_agent", route_to_agent)
    graph_builder.add_node("emotional_agent", emotional_agent)
    graph_builder.add_node("logical_agent", logical_agent)

    # Define edges
    graph_builder.add_edge(START, "classify_message")
    graph_builder.add_edge("classify_message", "route_to_agent")

    # Conditional routing based on agent decision
    graph_builder.add_conditional_edges(
        "route_to_agent",
        lambda state: state.get("next"),
        {
            "emotional_agent": "emotional_agent",
            "logical_agent": "logical_agent"
        }
    )

    # Both agents lead to END
    graph_builder.add_edge("emotional_agent", END)
    graph_builder.add_edge("logical_agent", END)

    # Compile the graph
    return graph_builder.compile()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def run_chatbot():
    """
    Main chatbot interface with user interaction loop.

    Features:
    - Interactive command-line interface
    - Automatic message classification and routing
    - Error handling and recovery
    - Session state management
    """
    # Build the conversation graph
    graph = build_chatbot_graph()

    # Initialize conversation state
    state = {"messages": [], "message_type": None}

    # Welcome message
    print("=" * 60)
    print("🤖 DUAL-AGENT CHATBOT")
    print("=" * 60)
    print("💝 Emotional Agent: Provides empathy and emotional support")
    print("🧠 Logical Agent: Provides facts and logical analysis")
    print()
    print("💬 Type your message and I'll route it to the right agent!")
    print("📝 Type 'exit', 'quit', or 'bye' to end the conversation")
    print("🔄 Type 'reset' to clear conversation history")
    print("=" * 60)
    print()

    # Example prompts
    print("💡 Example prompts:")
    print("   Emotional: 'I'm feeling overwhelmed with work'")
    print("   Logical: 'How does machine learning work?'")
    print()

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Thank you for chatting! Take care!")
                break
            elif user_input.lower() == 'reset':
                state = {"messages": [], "message_type": None}
                print("\n🔄 Conversation history cleared!\n")
                continue
            elif user_input.lower() in ['memory', 'history']:

                print("\n🔄 Conversation history!\n")
                print_history(state)
                continue
            elif not user_input:
                print("💭 Please enter a message.")
                continue

            # Add user message to state
            state["messages"] = state.get("messages", []) + [
                {"role": "user", "content": user_input}
            ]

            # Process through the graph
            print("\n🔄 Processing...")
            state = graph.invoke(state)

            # Display assistant response
            if state.get("messages") and len(state["messages"]) > 0:
                last_message = state["messages"][-1]

                # Handle both dict and AIMessage objects
                if hasattr(last_message, 'content'):  # AIMessage object
                    print(f"\nAssistant: {last_message.content}\n")
                    print("-" * 60)
                elif isinstance(last_message, dict) and last_message.get("role") == "assistant":  # Dict format
                    print(f"\nAssistant: {last_message['content']}\n")
                    print("-" * 60)
                else:
                    print("\n❌ No response generated. Please try again.\n")
            else:
                print("\n❌ No response generated. Please try again.\n")

        except KeyboardInterrupt:
            print("\n\n👋 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("🔄 Please try again or type 'reset' to start fresh.\n")

# =============================================================================
# DIAGNOSTICS AND TESTING
# =============================================================================

def test_classification():
    """Test both structured and manual classification methods."""
    print("🧪 Testing Classification Function")
    print("=" * 40)

    test_cases = [
        ("I'm feeling really sad today", "emotional"),
        ("How do computers work?", "logical"),
        ("My relationship is falling apart", "emotional"),
        ("What's the capital of France?", "logical"),
        ("I can't sleep because of anxiety", "emotional"),
        ("Explain quantum physics", "logical")
    ]

    for message, expected in test_cases:
        state = {"messages": [{"role": "user", "content": message}]}
        result = classify_message(state)
        actual = result["message_type"]
        status = "✅" if actual == expected else "❌"
        print(f"{status} '{message[:30]}...' → {actual} (expected: {expected})")

def show_graph_info():
    """Display information about the chatbot's architecture."""
    print("📊 CHATBOT ARCHITECTURE")
    print("=" * 50)
    print("🔄 Flow: START → Classifier → Router → Agent → END")
    print()
    print("🎯 Agents:")
    print("   💝 Emotional: Empathy, support, validation")
    print("   🧠 Logical: Facts, analysis, explanations")
    print()
    print("🔧 Technology Stack:")
    print("   • LangGraph: State management and routing")
    print("   • Ollama: Local LLM (llama3.2:1b)")
    print("   • Python: Core implementation")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Dual-Agent Chatbot...\n")

    # Check if user wants to run tests
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_classification()
            exit()
        elif sys.argv[1] == "info":
            show_graph_info()
            exit()

    # Run the main chatbot
    try:
        run_chatbot()
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure Ollama is running: ollama serve")
        print("   2. Ensure model is installed: ollama pull llama3.2:1b")
        print("   3. Check your internet connection")
        print("   4. Verify Python dependencies are installed")

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

"""
USAGE INSTRUCTIONS:

1. SETUP:
   pip install langchain-ollama langgraph python-dotenv
   ollama serve
   ollama pull llama3.2:1b

2. RUN:
   python chatbot.py          # Start interactive chat
   python chatbot.py test     # Test classification function
   python chatbot.py info     # Show architecture info

3. CHAT EXAMPLES:
   Emotional: "I'm stressed about my job interview tomorrow"
   Logical: "How does encryption work?"

4. COMMANDS:
   exit/quit/bye    - End conversation
   reset            - Clear history
   Ctrl+C           - Force quit

5. CUSTOMIZATION:
   - Modify system prompts in emotional_agent() and logical_agent()
   - Adjust classification logic in classify_message()
   - Change LLM parameters in the llm configuration
   - Add new agents by extending the graph structure
"""