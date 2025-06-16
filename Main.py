"""
Simplified Dual-Agent Chatbot using LangGraph and Ollama

This chatbot routes user messages to one of two specialized agents:
1. Emotional Agent: Provides empathy, emotional support, and therapeutic responses
2. Logical Agent: Provides factual information, analysis, and logical reasoning
3. Weather Agent: Provides weather details

Architecture:
    START → Classifier → Router → [Emotional Agent | Logical Agent | Weather Agent] → END

Requirements:
    - Ollama running locally with llama3.2:1b models
    - pip install langchain-ollama langgraph python-dotenv

Author: Mazhar Hassan
Date: 2025
"""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END


from agent_helpers import State, emotional_agent, logical_agent, phi3_llm
from classification_helper import classify_with_structured_output, classify_with_llm
from message_helper import print_history, extract_last_message
from agents.weather_agent import weather_agent

# Load environment variables
load_dotenv()


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def classify_message(state: State) -> dict:
    """
     Classify the user's message as 'emotional', 'logical', or 'weather' using structured output.

    Args:
        state: Current conversation state containing messages

    Returns:
        dict: Updated state with message_type classification

    Note:
        Uses Pydantic MessageClassifier for automatic validation and structured output.
        Falls back to manual parsing if structured output fails (for local models).
   """

    message_content = extract_last_message(state)

    try:
        #return classify_by_llm(logical_llm, message_content)
        return classify_with_structured_output(phi3_llm, message_content)
    except Exception as structured_error:
        print(f"⚠️ Structured output failed: {structured_error}")
        print("🔄 Falling back to manual parsing...")
        return classify_with_llm(phi3_llm, message_content)


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
    elif message_type == "weather":
        next_node = "weather_agent"
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
    graph_builder.add_node("weather_agent", weather_agent)

    # Define edges
    graph_builder.add_edge(START, "classify_message")
    graph_builder.add_edge("classify_message", "route_to_agent")

    # Conditional routing based on agent decision
    graph_builder.add_conditional_edges(
        "route_to_agent",
        lambda state: state.get("next"),
        {
            "weather_agent": "weather_agent",
            "emotional_agent": "emotional_agent",
            "logical_agent": "logical_agent",
        }
    )

    # Both agents lead to END
    graph_builder.add_edge("emotional_agent", END)
    graph_builder.add_edge("logical_agent", END)
    graph_builder.add_edge("weather_agent", END)

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
    print("🤖 Triple-AGENT CHATBOT")
    print("=" * 60)
    print("💝 Emotional Agent: Provides empathy and emotional support")
    print("🧠 Logical Agent: Provides facts and logical analysis")
    print("🌞 Weather Agent: Provides weather details")
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
        print("   2. Ensure models is installed: ollama pull llama3.2:1b")
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