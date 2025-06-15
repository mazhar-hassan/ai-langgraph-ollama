from pydantic import BaseModel, Field
from typing import Literal

from sympy import pretty_print


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class MessageClassifier(BaseModel):
    """
    Pydantic model for automatic message classification with validation.

    This ensures the LLM returns exactly one of the specified message types,
    with automatic validation and error handling.
    """
    message_type: Literal["emotional", "logical", "weather"] = Field(
        ...,
        description="Classify if the message requires an emotional, logical, or weather response."
    )


def classify_with_structured_output(llm, message_content):
    # Enhanced system prompt for three-way classification
    system_prompt = """You are a message classifier. Analyze the user's message and classify it as exactly one of:

        - "weather": if the message asks about weather, temperature, forecast, climate conditions, meteorology, or any weather-related queries (e.g., "What's the weather in London?", "Is it raining?", "Temperature today?")

        - "emotional": if the message asks for emotional support, therapy, deals with feelings, personal problems, relationships, stress, anxiety, sadness, happiness, mental health, or seeks empathy and understanding

        - "logical": if the message asks for facts, information, explanations, how-to guides, technical questions, analysis, calculations, or seeks objective knowledge (excluding weather)

        Respond with the classification in the exact format requested."""

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

        Definitions:
        - weather: The message is about weather, temperature, forecast, climate, or meteorology.
        - emotional: The message expresses feelings, distress, personal problems, relationships, mental health, anxiety, job loss, emotional struggles, etc.
        - logical: The message seeks facts, technical answers, explanations, or analysis — but not related to weather or emotions.

        Only return ONE WORD: weather, emotional, or logical. No extra text.

        Message: "{message_content}"
        Category:"""

        result = llm.invoke([{"role": "user", "content": classification_prompt}])

        classification = result.content.strip().lower()

        if classification == "emotional":
            message_type = "emotional"
        elif classification == "weather":
            message_type = "weather"
        else:
            message_type = "logical"

        print(f"🔍 Classification → '{message_content[:50]}...' → {message_type}")
        return {"message_type": message_type}

    except Exception as e:
        print(f"❌ LLM classification failed: {e}")
        print("🔄 Defaulting to logical agent...")
        return {"message_type": "logical"}
