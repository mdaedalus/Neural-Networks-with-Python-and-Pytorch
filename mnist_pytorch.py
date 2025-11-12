import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load MNIST Dataset
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)

# -----------------------------
# 2. Define Neural Network
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)

# -----------------------------
# 3. Define Loss and Optimizer
# -----------------------------
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# -----------------------------
# 4. Training Loop
# -----------------------------
epochs = 3
for epoch in range(epochs):
    for X, y in train_loader:
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# -----------------------------
# 5. Evaluate Accuracy
# -----------------------------
correct = 0
total = 0

with torch.no_grad():
    for X, y in test_loader:
        output = net(X.view(-1, 28*28))
        preds = torch.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.3f}")

# -----------------------------
# 6. Visualize a Sample
# -----------------------------
sample = X[0].view(28, 28)
plt.imshow(sample, cmap='gray')
plt.title(f"True Label: {y[0].item()}")
plt.show()

# -----------------------------
# 7. Inspect Network Prediction
# -----------------------------
reshaped_sample = X[0].view(-1, 28*28)
output = net(reshaped_sample)
pred_label = torch.argmax(output[0]).item()

print(f"Predicted Label: {pred_label}")
print(f"Raw Network Output: {output[0]}")
