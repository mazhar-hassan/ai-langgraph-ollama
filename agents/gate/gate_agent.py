from agents.gate.train_model import LogicGateModel
import torch
import joblib
from pathlib import Path

# 4. INFERENCE WRAPPER
class LogicGatePredictor:
    def __init__(self):
        # Get directory where THIS file (gate_agent.py) is located
        file_dir = Path(__file__).parent

        # Build absolute paths relative to this file
        self.model_path = file_dir / "models" / "logic_gate_model.pth"
        self.info_path = file_dir / "models" / "logic_gate_info.pkl"

        # Convert to strings for compatibility
        self.model_path = str(self.model_path)
        self.info_path = str(self.info_path)

        self.model = LogicGateModel()
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        self.info = joblib.load(self.info_path)

    def predict(self, x, y, z):
        """
        Predict logic gate output
        x, y: binary inputs (0 or 1)
        z: gate type (0 = AND, 1 = OR)
        """
        with torch.no_grad():
            input_tensor = torch.FloatTensor([[x, y, z]])
            output = self.model(input_tensor)
            prediction = (output > 0.5).item()
            confidence = output.item()

            gate_type = self.info['gate_types'][z]

            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'gate_type': gate_type,
                'inputs': {'x': x, 'y': y},
                'expected': x & y if z == 0 else x | y
            }

    def batch_predict(self, inputs):
        """Predict for multiple inputs"""
        results = []
        for x, y, z in inputs:
            results.append(self.predict(x, y, z))
        return results


# 5. INTEGRATION WITH EXISTING CHATBOT
def create_logic_gate_agent(predictor):
    """Create a logic gate agent for the chatbot"""

    def logic_gate_agent(state):
        last_message = state["messages"][-1]

        try:
            # Parse the message to extract x, y, z
            # This is a simple parser - you can make it more sophisticated
            content = last_message.content.lower()

            # Look for patterns like "x=1 y=0 z=1" or "1 and 0 using or gate"
            if "x=" in content and "y=" in content and "z=" in content:
                # Extract values using simple parsing
                x = int(content.split("x=")[1].split()[0])
                y = int(content.split("y=")[1].split()[0])
                z = int(content.split("z=")[1].split()[0])
            else:
                # Try to parse natural language
                # This is where you'd add more sophisticated NLP
                return {"messages": [{"role": "assistant",
                                      "content": "Please provide inputs in format: x=0 y=1 z=0 (where z: 0=AND, 1=OR)"}]}

            # Validate inputs
            if not all(val in [0, 1] for val in [x, y, z]):
                return {"messages": [{"role": "assistant",
                                      "content": "All inputs (x, y, z) must be 0 or 1"}]}

            # Get prediction
            result = predictor.predict(x, y, z)

            # Format response
            response = f"""🔧 Logic Gate Calculation:

📥 Inputs: x={x}, y={y}
🚪 Gate Type: {result['gate_type']} (z={z})
📤 Output: {result['prediction']}
🎯 Confidence: {result['confidence']:.3f}
✅ Expected: {result['expected']}

Logic: {x} {result['gate_type']} {y} = {result['prediction']}"""

            return {"messages": [{"role": "assistant", "content": response}]}

        except Exception as e:
            return {"messages": [{"role": "assistant",
                                  "content": f"Error processing logic gate request: {str(e)}"}]}

    return logic_gate_agent


def create_custom_gate_agent():
    return create_logic_gate_agent(LogicGatePredictor())

if __name__ == "__main__":
    # Test the predictor
    print("\n🧪 Testing predictor...")
    predictor = LogicGatePredictor()

    # Test cases
    test_cases = [
        (0, 0, 0),  # 0 AND 0 = 0
        (1, 1, 0),  # 1 AND 1 = 1
        (0, 1, 1),  # 0 OR 1 = 1
        (0, 0, 1),  # 0 OR 0 = 0
    ]

    for x, y, z in test_cases:
        result = predictor.predict(x, y, z)
        print(f"Input: x={x}, y={y}, z={z} ({result['gate_type']}) -> "
              f"Prediction: {result['prediction']}, Expected: {result['expected']}")
