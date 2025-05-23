import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Definicja prostej sieci neuronowej
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Warstwa wejściowa
        self.fc2 = nn.Linear(128, 10)   # Warstwa wyjściowa
        self.relu = nn.ReLU()           # Funkcja aktywacji

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicjalizacja modelu
model = SimpleNN()

# Definicja funkcji kosztu i optymalizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Wczytanie danych MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=32, shuffle=True)

# Trening modelu
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs.view(-1, 784))
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Zapisanie modelu lokalnie
torch.save(model.state_dict(), 'simple_nn.pth')
print("Model zapisany lokalnie jako 'simple_nn.pth'")
