import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Input layer to hidden layer
        self.fc2 = nn.Linear(5, 1)    # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)
        return x

# Create the model
model = SimpleNN()

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input and target
input_data = torch.randn(1, 10)
target = torch.tensor([[1.0]])

# Forward pass
output = model(input_data)
loss = criterion(output, target)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Output: {output}, Loss: {loss.item()}")
