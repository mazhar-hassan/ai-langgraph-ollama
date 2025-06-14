from dotenv import load_dotenv
from typing import Annotated, Literal, Union
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# Initialize Ollama with llama3.2:1b model
llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7
)


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: Union[str, None]


def classify_message(state: State):
    last_message = state["messages"][-1]

    # For Ollama, we'll use a simpler approach since structured output might not work as reliably
    classification_prompt = f"""
    Classify the following user message as either 'emotional' or 'logical':

    - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
    - 'logical': if it asks for facts, information, logical analysis, or practical solutions

    User message: "{last_message.content}"

    Respond with only one word: either "emotional" or "logical"
    """

    result = llm.invoke([{"role": "user", "content": classification_prompt}])

    # Extract the classification from the response
    classification = result.content.strip().lower()
    if "emotional" in classification:
        message_type = "emotional"
    else:
        message_type = "logical"

    return {"message_type": message_type}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}


def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}

    print("Chatbot initialized with Ollama llama3.2:1b")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Message: ")
        if user_input.lower() == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        try:
            state = graph.invoke(state)

            if state.get("messages") and len(state["messages"]) > 0:
                last_message = state["messages"][-1]
                print(f"Assistant: {last_message.content}\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:1b model is installed\n")


if __name__ == "__main__":
    run_chatbot()