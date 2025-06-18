from llm_definition import State


def router_agent(state: State) -> dict:
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
    elif message_type == "logic_gate":
        next_node = "logic_gate_agent"
    else:
        next_node = "logical_agent"

    print(f"🔀 Routing to: {next_node}")
    return {"next": next_node}
