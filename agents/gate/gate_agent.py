from agent_helpers import phi3_llm, lama3_2_llm
from agents.gate.train_model import LogicGateModel
import torch
import joblib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class LogicGateParams(BaseModel):
    """Structured output model for logic gate parameters."""
    x: int = Field(..., description="First input value (0 or 1)")
    y: int = Field(..., description="Second input value (0 or 1)")
    z: int = Field(..., description="Gate type (0 for AND, 1 for OR)")
    confidence: str = Field(..., description="Confidence level: high, medium, or low")
    reasoning: str = Field(..., description="Brief explanation of how values were extracted")


class ParseResult(BaseModel):
    """Result of parsing attempt."""
    success: bool = Field(..., description="Whether parsing was successful")
    error_message: Optional[str] = Field(None, description="Error message if parsing failed")


# =============================================================================
# AGENT 1: LLM-POWERED PARSER AGENT
# =============================================================================

class LLMLogicGateParserAgent:
    """
    First agent in the chain: Uses LLM to extract x, y, z values from natural language.
    Much more robust than regex-based parsing.
    """

    def __init__(self, primary_llm=None, fallback_llm=None):
        self.name = "LLM Logic Gate Parser"
        self.description = "Uses LLM to extract logic gate parameters from any input format"

        # Use provided LLMs or defaults
        self.primary_llm = primary_llm or phi3_llm
        self.fallback_llm = fallback_llm or lama3_2_llm

        print(f"✅ {self.name} initialized with LLM support")

    def create_parsing_prompt(self, user_input: str) -> str:
        """Create a detailed prompt for parameter extraction."""
        return f"""You are an expert at extracting logic gate parameters from text.

TASK: Extract x, y, and z values from the user's input.

RULES:
- x and y are binary inputs (must be 0 or 1)
- z is gate type (0 = AND gate, 1 = OR gate)
- If any value is unclear or missing, indicate low confidence

EXAMPLES:
Input: "x=0 y=1 z=0"
Output: x=0, y=1, z=0 (formal format)

Input: "if x is zero, and y is 1 where gate is AND"
Output: x=0, y=1, z=0 (AND gate means z=0)

Input: "what is 1 OR 0?"
Output: x=1, y=0, z=1 (OR gate means z=1)

Input: "0 and 1 with AND gate"
Output: x=0, y=1, z=0 (first number is x, second is y)

USER INPUT: "{user_input}"

Extract the values and respond in this EXACT JSON format:
{{
    "x": <number>,
    "y": <number>, 
    "z": <number>,
    "confidence": "<high/medium/low>",
    "reasoning": "<brief explanation>"
}}

JSON Response:"""

    def parse_with_structured_output(self, user_input: str, llm) -> Optional[Dict]:
        """Try parsing with structured output (Pydantic)."""
        try:
            # Create LLM with structured output
            structured_llm = llm.with_structured_output(LogicGateParams)

            # Create prompt
            prompt = self.create_parsing_prompt(user_input)

            # Get structured response
            result = structured_llm.invoke([{"role": "user", "content": prompt}])

            # Validate the values
            if (result.x in [0, 1] and result.y in [0, 1] and result.z in [0, 1]):
                return {
                    "x": result.x,
                    "y": result.y,
                    "z": result.z,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "method": "structured_output"
                }
            else:
                print(f"⚠️ Invalid values from structured output: x={result.x}, y={result.y}, z={result.z}")
                return None

        except Exception as e:
            print(f"⚠️ Structured output failed: {e}")
            return None

    def parse_with_json_extraction(self, user_input: str, llm) -> Optional[Dict]:
        """Fallback: Parse JSON from raw LLM response."""
        try:
            prompt = self.create_parsing_prompt(user_input)

            # Get raw response
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()

            print(f"🔍 Raw LLM response: {response_text[:100]}...")

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)

                # Validate and convert
                x = int(parsed_data.get('x', -1))
                y = int(parsed_data.get('y', -1))
                z = int(parsed_data.get('z', -1))

                if x in [0, 1] and y in [0, 1] and z in [0, 1]:
                    return {
                        "x": x,
                        "y": y,
                        "z": z,
                        "confidence": parsed_data.get('confidence', 'medium'),
                        "reasoning": parsed_data.get('reasoning', 'Extracted from LLM response'),
                        "method": "json_extraction"
                    }
                else:
                    print(f"⚠️ Invalid values from JSON: x={x}, y={y}, z={z}")
                    return None
            else:
                print("⚠️ No valid JSON found in response")
                return None

        except Exception as e:
            print(f"⚠️ JSON extraction failed: {e}")
            return None

    def parse_with_simple_extraction(self, user_input: str, llm) -> Optional[Dict]:
        """Last resort: Simple keyword-based extraction guided by LLM."""
        try:
            simple_prompt = f"""Extract the three numbers from this logic gate query:

User input: "{user_input}"

Find:
1. First input value (x): should be 0 or 1
2. Second input value (y): should be 0 or 1  
3. Gate type (z): 0 for AND gate, 1 for OR gate

Respond with just three numbers separated by spaces like: 0 1 0"""

            response = llm.invoke([{"role": "user", "content": simple_prompt}])
            response_text = response.content.strip()

            print(f"🔍 Simple extraction response: {response_text}")

            # Extract numbers from response
            numbers = []
            for char in response_text:
                if char in '01':
                    numbers.append(int(char))
                    if len(numbers) == 3:
                        break

            if len(numbers) == 3:
                return {
                    "x": numbers[0],
                    "y": numbers[1],
                    "z": numbers[2],
                    "confidence": "low",
                    "reasoning": "Simple extraction fallback",
                    "method": "simple_extraction"
                }
            else:
                return None

        except Exception as e:
            print(f"⚠️ Simple extraction failed: {e}")
            return None

    def process(self, message: str) -> Dict[str, Any]:
        """
        Main processing method using cascading LLM approaches.

        Args:
            message: User input message

        Returns:
            Dict with parsing results and status
        """
        print(f"🔍 LLM Parser processing: '{message[:50]}...'")

        # Strategy 1: Try structured output with primary LLM
        result = self.parse_with_structured_output(message, self.primary_llm)
        if result:
            print(f"✅ Structured output successful with primary LLM")
            return {
                "success": True,
                "llm_used": "primary_structured",
                **result
            }

        # Strategy 2: Try JSON extraction with primary LLM
        result = self.parse_with_json_extraction(message, self.primary_llm)
        if result:
            print(f"✅ JSON extraction successful with primary LLM")
            return {
                "success": True,
                "llm_used": "primary_json",
                **result
            }

        # Strategy 3: Try structured output with fallback LLM
        result = self.parse_with_structured_output(message, self.fallback_llm)
        if result:
            print(f"✅ Structured output successful with fallback LLM")
            return {
                "success": True,
                "llm_used": "fallback_structured",
                **result
            }

        # Strategy 4: Try JSON extraction with fallback LLM
        result = self.parse_with_json_extraction(message, self.fallback_llm)
        if result:
            print(f"✅ JSON extraction successful with fallback LLM")
            return {
                "success": True,
                "llm_used": "fallback_json",
                **result
            }

        # Strategy 5: Simple extraction with primary LLM
        result = self.parse_with_simple_extraction(message, self.primary_llm)
        if result:
            print(f"✅ Simple extraction successful with primary LLM")
            return {
                "success": True,
                "llm_used": "primary_simple",
                **result
            }

        # Strategy 6: Simple extraction with fallback LLM
        result = self.parse_with_simple_extraction(message, self.fallback_llm)
        if result:
            print(f"✅ Simple extraction successful with fallback LLM")
            return {
                "success": True,
                "llm_used": "fallback_simple",
                **result
            }

        # All strategies failed
        print("❌ All LLM parsing strategies failed")
        return {
            "success": False,
            "llm_used": "none",
            "x": None,
            "y": None,
            "z": None,
            "method": None,
            "message": "Could not extract logic gate parameters. Please be more specific about x, y values and gate type (AND/OR)."
        }


# =============================================================================
# AGENT 2: LOGIC GATE PREDICTOR AGENT (UNCHANGED)
# =============================================================================

class LogicGatePredictor:
    """
    Second agent in the chain: Makes predictions using the trained model.
    Takes extracted parameters and returns detailed results.
    """

    def __init__(self):
        self.name = "Logic Gate Predictor"
        self.description = "Makes predictions using trained neural network"

        # Get directory where THIS file is located
        file_dir = Path(__file__).parent

        # Build absolute paths relative to this file
        self.model_path = file_dir / "models" / "logic_gate_model.pth"
        self.info_path = file_dir / "models" / "logic_gate_info.pkl"

        # Convert to strings for compatibility
        self.model_path = str(self.model_path)
        self.info_path = str(self.info_path)

        # Load model and info
        self.model = LogicGateModel()
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        self.info = joblib.load(self.info_path)

    def predict(self, x: int, y: int, z: int) -> Dict[str, Any]:
        """
        Predict logic gate output using the trained model.
        """
        # Validate inputs
        if not all(val in [0, 1] for val in [x, y, z]):
            raise ValueError("All inputs (x, y, z) must be 0 or 1")

        with torch.no_grad():
            input_tensor = torch.FloatTensor([[x, y, z]])
            output = self.model(input_tensor)
            prediction = (output > 0.5).item()
            confidence = output.item()

            gate_type = self.info['gate_types'][z]

            # Calculate expected result for verification
            expected = x & y if z == 0 else x | y

            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'gate_type': gate_type,
                'inputs': {'x': x, 'y': y},
                'expected': expected,
                'correct': int(prediction) == expected
            }


# =============================================================================
# AGENT 3: ENHANCED RESPONSE FORMATTER AGENT
# =============================================================================

class LLMLogicGateResponseAgent:
    """
    Third agent in the chain: Formats responses with LLM awareness.
    """

    def __init__(self):
        self.name = "LLM Logic Gate Response Formatter"
        self.description = "Formats prediction results with LLM parsing details"

    def format_success_response(self, parse_result: Dict, prediction_result: Dict) -> str:
        """Format a successful prediction response with LLM details."""
        x = parse_result['x']
        y = parse_result['y']
        z = parse_result['z']

        gate_type = prediction_result['gate_type']
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        expected = prediction_result['expected']
        correct = prediction_result['correct']

        # Create the main response
        response = f"""🔧 **Logic Gate Calculation:**

📥 **Inputs:** x={x}, y={y}
🚪 **Gate Type:** {gate_type} (z={z})
📤 **Output:** {prediction}
🎯 **Neural Net Confidence:** {confidence:.3f}
✅ **Expected:** {expected}
{'✅ **Status:** Correct!' if correct else '❌ **Status:** Incorrect!'}

💡 **Logic:** {x} {gate_type} {y} = {prediction}"""

        # Add LLM parsing details
        if 'llm_used' in parse_result:
            llm_info = parse_result['llm_used']
            parsing_confidence = parse_result.get('confidence', 'unknown')
            reasoning = parse_result.get('reasoning', 'No reasoning provided')

            response += f"""

🧠 **LLM Parsing Details:**
• **Method:** {llm_info}
• **Parse Confidence:** {parsing_confidence}
• **LLM Reasoning:** {reasoning}"""

        return response

    def format_error_response(self, parse_result: Dict, error: str = None) -> str:
        """Format an error response with LLM troubleshooting."""
        base_error = f"❌ **LLM Parsing Failed:** {parse_result['message']}"

        if error:
            base_error += f"\n\n**Additional Error:** {error}"

        examples = """

🤖 **The LLM tried multiple strategies but couldn't extract values.**

📝 **Please try being more explicit:**

**Clear Examples:**
• "x equals 0, y equals 1, use AND gate"
• "first input is 1, second input is 0, gate type is OR"
• "x=0 y=1 z=0"
• "calculate 1 AND 0"

**Gate Types:**
• Use "AND" or "z=0" for AND gates
• Use "OR" or "z=1" for OR gates

🧠 **The LLM understands natural language better when you're specific about which value is x, which is y, and what gate type to use.**"""

        return base_error + examples


# =============================================================================
# ENHANCED CHAIN COORDINATOR WITH LLM POWER
# =============================================================================

class LLMLogicGateChain:
    """
    Coordinates the chain of agents with LLM-powered parsing.
    """

    def __init__(self, primary_llm=None, fallback_llm=None):
        self.parser = LLMLogicGateParserAgent(primary_llm, fallback_llm)
        self.predictor = LogicGatePredictor()
        self.formatter = LLMLogicGateResponseAgent()
        print("✅ LLM-Powered Logic Gate Agent Chain initialized successfully")

    def process(self, message: str) -> str:
        """
        Process user message through the LLM-powered agent chain.
        """
        try:
            # Step 1: LLM-powered parsing
            print(f"🤖 LLM Parser Agent processing: '{message[:50]}...'")
            parse_result = self.parser.process(message)

            if not parse_result['success']:
                # LLM parsing failed - return helpful error message
                return self.formatter.format_error_response(parse_result)

            # Step 2: Neural network prediction
            print(f"🧠 Predictor Agent processing: x={parse_result['x']}, y={parse_result['y']}, z={parse_result['z']}")
            prediction_result = self.predictor.predict(
                parse_result['x'],
                parse_result['y'],
                parse_result['z']
            )

            # Step 3: Enhanced response formatting
            print(
                f"📝 Response Agent formatting result: {prediction_result['gate_type']} = {prediction_result['prediction']}")
            final_response = self.formatter.format_success_response(parse_result, prediction_result)

            return final_response

        except Exception as e:
            error_msg = f"LLM chain processing error: {str(e)}"
            print(f"❌ {error_msg}")
            return self.formatter.format_error_response(
                {"success": False, "message": "LLM processing failed"},
                error=str(e)
            )


# =============================================================================
# INTEGRATION WITH EXISTING CHATBOT
# =============================================================================

def create_llm_logic_gate_agent(primary_llm=None, fallback_llm=None):
    """
    Create an LLM-powered logic gate agent for the chatbot.

    Args:
        primary_llm: Primary LLM for parsing (default: llama3.2:1b)
        fallback_llm: Fallback LLM for parsing (default: phi3:mini)
    """

    # Use provided LLMs or defaults
    fallback = fallback_llm or lama3_2_llm
    primary = primary_llm or phi3_llm

    # Initialize the LLM-powered agent chain
    gate_chain = LLMLogicGateChain(primary, fallback)

    def logic_gate_agent(state):
        """
        Chatbot agent function that uses LLM-powered parsing.
        """
        try:
            # Get the last message
            last_message = state["messages"][-1]

            # Extract content based on message type
            if hasattr(last_message, 'content'):
                content = last_message.content
            elif isinstance(last_message, dict):
                content = last_message.get("content", "")
            else:
                content = str(last_message)

            # Process through the LLM-powered agent chain
            gate_response = gate_chain.process(content)

            return {
                "messages": [{"role": "assistant", "content": gate_response}]
            }

        except Exception as e:
            error_response = f"❌ LLM Logic gate agent error: {str(e)}\n\n🤖 The AI models may be unavailable. Please ensure Ollama is running with llama3.2:1b and phi3:mini models."

            return {
                "messages": [{"role": "assistant", "content": error_response}]
            }

    return logic_gate_agent


# Backward compatibility
def create_custom_gate_agent():
    """Backward compatibility function that now uses LLM parsing."""
    return create_llm_logic_gate_agent()


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the LLM-powered chained agents
    print("🧪 Testing LLM-Powered Logic Gate Agents")
    print("=" * 50)

    try:
        chain = LLMLogicGateChain()

        test_inputs = [
            "if x is zero, and y is 1 where gate is AND gate",
            "what happens when first input is 1 and second input is 0 with OR gate?",
            "x=1 y=0 z=0",
            "what is 1 OR 0?",
            "calculate 0 AND 1",
            "first bit is 1, second bit is 1, use AND logic",
            "can you compute one or zero?",
            "invalid input test without clear parameters"
        ]

        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n{i}. Testing: '{test_input}'")
            print("-" * 50)
            response = chain.process(test_input)
            print(response)
            print()

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("Make sure Ollama is running with llama3.2:1b and phi3:mini models!")