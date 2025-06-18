from llm_definition import State, phi3_llm
from message_helper import create_message_with_history


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

