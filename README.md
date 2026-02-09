# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="869" height="713" alt="Screenshot 2026-02-06 184805" src="https://github.com/user-attachments/assets/6e808709-6ddc-4642-906f-b2f4c05dbc23" />

## DESIGN STEPS:


### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: ARUN KUMAR B
### Register Number: 212223230021

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```



## Dataset Information

<img width="1149" height="465" alt="image" src="https://github.com/user-attachments/assets/a666efea-0f91-4259-b67e-70dcd7fd2e0c" />


## OUTPUT



### Confusion Matrix

<img width="699" height="611" alt="image" src="https://github.com/user-attachments/assets/b92e2c6e-e858-4ad7-9323-4ae261214365" />


### Classification Report

<img width="571" height="241" alt="image" src="https://github.com/user-attachments/assets/a6895600-e11b-4762-b606-e84b2e740420" />



### New Sample Data Prediction

<img width="779" height="258" alt="image" src="https://github.com/user-attachments/assets/726a289f-2ecf-4c55-bb91-cf198b4da60c" />

## RESULT

Thus the neural network classification model was successfully developed.
