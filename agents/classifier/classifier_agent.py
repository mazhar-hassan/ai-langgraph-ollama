from llm_definition import State, phi3_llm
from message_helper import extract_last_message
from pydantic import BaseModel, Field
from typing import Literal

# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class MessageClassifier(BaseModel):
    """
    Pydantic models for automatic message classification with validation.

    This ensures the LLM returns exactly one of the specified message types,
    with automatic validation and error handling.
    """
    message_type: Literal["emotional", "logical", "weather", "logic_gate"] = Field(
        ...,
        description="Classify if the message requires emotional, logical, weather, or logic gate response."
    )

# =============================================================================
# CLASSIFIER Agent
# =============================================================================


def classifier_agent(state: State) -> dict:
    """
     Classify the user's message as 'emotional', 'logical', 'weather', or 'gate' using structured output.

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

def classify_with_structured_output(llm, message_content):
    # Enhanced system prompt for three-way classification
    system_prompt = """You are a message classifier. Analyze the user's message and classify it as exactly one of:

        - "weather": if the message asks about weather, temperature, forecast, climate conditions, meteorology, or any weather-related queries (e.g., "What's the weather in London?", "Is it raining?", "Temperature today?")
        - "emotional": if the message asks for emotional support, therapy, deals with feelings, personal problems, relationships, stress, anxiety, sadness, happiness, mental health, or seeks empathy and understanding
        - "logical": if the message asks for facts, information, explanations, how-to guides, technical questions, analysis, calculations, or seeks objective knowledge (excluding weather)
        - 'logic_gate': logic gates, boolean operations, AND/OR operations, binary calculations, x/y/z inputs

        Respond with only one word: either "emotional", "logical", "weather", or "logic_gate"
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Classify this message: {message_content}"}
    ]

    classifier_llm = llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke(messages)

    print(f"🔍 Classification (Structured): '{message_content[:50]}...' → {result.message_type}")
    return {"message_type": result.message_type}


def classify_with_llm(llm, message_content):
    try:
        classification_prompt = f"""
        You are a strict classifier. Classify the following user message into exactly one of these categories:

        - weather
        - emotional
        - logical
        - logic_gate

        Definitions:
        - weather: The message is about weather, temperature, forecast, climate, or meteorology.
        - emotional: The message expresses feelings, distress, personal problems, relationships, mental health, anxiety, job loss, emotional struggles, etc.
        - logical: The message seeks facts, technical answers, explanations, or analysis — but not related to weather or emotions.
        - logic_gate: logic gates, boolean operations, AND/OR operations, binary calculations, x/y/z inputs

        Respond with only one word: either "emotional", "logical", "weather", or "logic_gate"

        Message: "{message_content}"
        Category:"""

        result = llm.invoke([{"role": "user", "content": classification_prompt}])

        classification = result.content.strip().lower()

        if classification == "emotional":
            message_type = "emotional"
        elif classification == "weather":
            message_type = "weather"
        elif "logic_gate" in classification:
            message_type = "logic_gate"
        else:
            message_type = "logical"

        print(f"🔍 Classification → '{message_content[:50]}...' → {message_type}")
        return {"message_type": message_type}

    except Exception as e:
        print(f"❌ LLM classification failed: {e}")
        print("🔄 Defaulting to logical agent...")
        return {"message_type": "logical"}


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
        result = classifier_agent(state)
        actual = result["message_type"]
        status = "✅" if actual == expected else "❌"
        print(f"{status} '{message[:30]}...' → {actual} (expected: {expected})")
