from langgraph.graph import StateGraph, START, END

from agents.classifier.classifier_agent import classifier_agent
from agents.emotional_agent import emotional_agent
from agents.gate.gate_agent import create_custom_gate_agent
from agents.logical_agent import logical_agent
from agents.router_agent import router_agent
from agents.weather_agent import weather_agent
from llm_definition import State


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_chatbot_graph() -> StateGraph:
    """
    Construct the conversation flow graph.

    Flow:
        START → classify_message → route_to_agent → [emotional_agent | logical_agent | weather_agent | logica_gate_agent] → END

    Returns:
        StateGraph: Compiled conversation graph
    """
    # Initialize graph builder
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("classify_message", classifier_agent)
    graph_builder.add_node("route_to_agent", router_agent)
    graph_builder.add_node("emotional_agent", emotional_agent)
    graph_builder.add_node("logical_agent", logical_agent)
    graph_builder.add_node("weather_agent", weather_agent)
    graph_builder.add_node("logic_gate_agent", create_custom_gate_agent())

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
            "logic_gate_agent":"logic_gate_agent"
        }
    )

    # Both agents lead to END
    graph_builder.add_edge("emotional_agent", END)
    graph_builder.add_edge("logical_agent", END)
    graph_builder.add_edge("weather_agent", END)
    graph_builder.add_edge("logic_gate_agent", END)

    # Compile the graph
    return graph_builder.compile()


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

