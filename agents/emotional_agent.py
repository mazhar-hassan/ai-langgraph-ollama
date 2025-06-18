from llm_definition import State, lama3_2_llm
from message_helper import create_message_with_history


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
