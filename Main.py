"""
Simplified Dual-Agent Chatbot using LangGraph and Ollama

This chatbot routes user messages to one of two specialized agents:
1. Emotional Agent: Provides empathy, emotional support, and therapeutic responses
2. Logical Agent: Provides factual information, analysis, and logical reasoning
3. Weather Agent: Provides weather details
3. Logic Gate Agent: Provides logic operation of Gates (AND and OR)

Architecture:
    START → Classifier → Router → [Emotional Agent | Logical Agent | Weather Agent | Gate Agent] → END

Requirements:
    - Ollama running locally with llama3.2:1b models
    - pip install -r requirements.txt

Author: Mazhar Hassan
Date: 2025
"""

from dotenv import load_dotenv

from agents.classifier.classifier_agent import test_classification
from langgraph_helper import build_chatbot_graph, show_graph_info
from message_helper import print_history

# Load environment variables
load_dotenv()

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
    print("🤖 Quad-AGENT CHATBOT (4 Agents)")
    print("=" * 60)
    print("💝 Emotional Agent: Provides empathy and emotional support")
    print("🧠 Logical Agent: Provides facts and logical analysis")
    print("🌞 Weather Agent: Provides weather details")
    print("🧮 Gate Agent: Provides operation results of AND and OR Gate")
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

