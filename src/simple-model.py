import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json

# Set the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your data
with open("datasets/train.json", "r") as f:
    data = json.load(f)

# Extracting features and labels
questions_masked = [d["question_mask"] for d in data]
question = [d["question"] for d in data]
# target labels should be the number that is masked in the question
# try to find number in the question
target_labels = []
found_num = []
for q in question:
    num_in_q = [num for num in q.split() if num.isdigit()]
    found_num = found_num + num_in_q

for num in found_num:
    target_labels.append(float(num))
# Preprocessing text
def preprocess_text(text):
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

questions_masked = [preprocess_text(question) for question in questions_masked]

# Tokenizing text
word_index = {word: idx + 1 for idx, word in enumerate(set(" ".join(questions_masked).split()))}
sequences = [[word_index[word] for word in question.split()] for question in questions_masked]

# Padding sequences
max_len = max(len(seq) for seq in sequences)
padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in sequences]

# Convert to PyTorch tensors and move to the device
X = torch.tensor(padded_sequences, dtype=torch.long).to(device)
y = torch.tensor(target_labels, dtype=torch.float32).unsqueeze(1).to(device)

# Define the model and move it to the device
class FillNumModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(FillNumModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(max_len * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the device
model = FillNumModel(vocab_size=len(word_index) + 1, embedding_dim=2048, hidden_dim=1024, output_dim=1).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100  # Increase the number of epochs for better convergence
batch_size = 1

for epoch in range(epochs):
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

print("Finished training!")
# Testing the model
with torch.no_grad():
    model.eval()
    idx = 0
    X_test = X[idx].unsqueeze(0)
    y_test = y[idx].unsqueeze(0)
    pred = model(X_test)
    print(f"Predicted: {pred.item():.4f}, Target: {y_test.item():.4f}")