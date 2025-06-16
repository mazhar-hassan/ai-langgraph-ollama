import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


# 1. DATA GENERATION
def generate_logic_gate_data(num_samples=10000):
    """
    Generate training data for logic gates
    x, y: binary inputs (0 or 1)
    z: gate type (0 = AND, 1 = OR)
    output: result of the operation
    """
    data = []
    labels = []

    for _ in range(num_samples):
        x = np.random.randint(0, 2)  # 0 or 1
        y = np.random.randint(0, 2)  # 0 or 1
        z = np.random.randint(0, 2)  # 0 = AND, 1 = OR

        # Calculate output based on gate type
        if z == 0:  # AND gate
            output = x & y
        else:  # OR gate (z == 1)
            output = x | y

        data.append([x, y, z])
        labels.append(output)

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)


# 2. NEURAL NETWORK MODEL
class LogicGateModel(nn.Module):
    def __init__(self):
        super(LogicGateModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),  # Input: x, y, z
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),  # Output: binary result
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        return self.network(x)


# 3. TRAINING FUNCTION
def train_logic_gate_model():
    """Train the logic gate models"""
    print("🔧 Generating training data...")
    X, y = generate_logic_gate_data(10000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    print("🧠 Initializing models...")
    model = LogicGateModel()
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training models...")
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

    # Evaluate models
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted = (test_outputs > 0.5).float()
        accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        print(f"🎯 Test Accuracy: {accuracy:.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/logic_gate_model.pth")

    # Save some metadata
    model_info = {
        'input_features': ['x', 'y', 'z'],
        'output': 'binary_result',
        'gate_types': {0: 'AND', 1: 'OR'},
        'accuracy': accuracy
    }
    joblib.dump(model_info, "models/logic_gate_info.pkl")

    print("✅ Model saved successfully!")
    return model, model_info




# 6. MAIN TRAINING SCRIPT
if __name__ == "__main__":
    print("🚀 Starting Logic Gate Model Training...")

    # Train the models
    model, info = train_logic_gate_model()


    print("\n✅ Training and testing completed!")
    print("📁 Model saved in 'models/' directory")
    print("🔗 Ready to integrate with chatbot!")