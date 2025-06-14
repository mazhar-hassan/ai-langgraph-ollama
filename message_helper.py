from langchain_core.messages import BaseMessage

def normalize_messages(messages):
    """Converts all messages to OpenAI-style dicts."""
    normalized = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            role = (
                "user" if msg.__class__.__name__ == "HumanMessage" else
                "assistant" if msg.__class__.__name__ == "AIMessage" else
                "system"
            )
            normalized.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            normalized.append(msg)
        else:
            raise TypeError(f"Unknown message type: {type(msg)}")
    return normalized

def create_message_with_history(state, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    history = normalize_messages(state.get("messages", [])[-20:])
    messages.extend(history)
    ## Following is not needed as it's already added to the state messages list
    #messages.append({"role": "user", "content": message_content})
    return messages

# Extract content from message (handle both dict and AIMessage objects)
def extract_last_message(state):
    last_message = state["messages"][-1]
    # Extract content from message (handle both dict and AIMessage objects)
    if hasattr(last_message, 'content'):  # AIMessage object
        message_content = last_message.content
    elif isinstance(last_message, dict):  # Dict format
        message_content = last_message.get("content", "")
    else:
        message_content = str(last_message)

    return message_content


def print_history(state):
    print("*" * 60)
    for msg in state.get("messages", []):
        # Support both plain dicts and LangChain message objects
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "unknown")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        print(f"[{role}] {content}")
        print("-" * 100)
    print("*" * 60)