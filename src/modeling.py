import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

 #Model Selection-Choose a machine learning model (e.g., Random Forest, Neural Network


#Create a simple neural network using PyTorch
class PredictiveModel(nn.Module):
    def __init__(self, input_size):
        super(PredictiveModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Assuming binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train_model(X_train, y_train, input_size, epochs=100, lr=0.001):
    model = PredictiveModel(input_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train.values, dtype=torch.float32)
        labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model



    
   

    