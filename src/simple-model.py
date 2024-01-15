import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm

# Sample data
with open("datasets/train.json", "r") as f:
    data = json.load(f)

# select only the first 1000 samples
data = data[:1000]

should_train = True


# Define the model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Preprocess the data
max_len = 128


class CustomDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        masked_number = torch.tensor(float(item['number']))  # Masked number as input
        actual_number = torch.tensor(float(item['number']))  # Actual number as target
        return {
            'masked_number': masked_number,
            'actual_number': actual_number
        }


# Create datasets and dataloader
dataset = CustomDataset(data, max_len)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define the model, loss function, and optimizer
input_size = 1
hidden_size = 64
output_size = 1

model = SimpleModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if should_train:
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        total_loss = 0.0

        for batch in progress_bar:
            masked_number = batch['masked_number'].to(device)
            actual_number = batch['actual_number'].to(device)

            # Forward pass
            predicted_number = model(masked_number.unsqueeze(1))

            # Calculate the loss using Mean Squared Error
            loss = criterion(predicted_number, actual_number.unsqueeze(1))

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            progress_bar.set_postfix({'Loss': total_loss / (len(progress_bar) + 1e-12)})

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'simple_model.pth')

# Inference
# Load the saved model
loaded_model = SimpleModel(input_size, hidden_size, output_size)
loaded_model.load_state_dict(torch.load('simple_model.pth'))
loaded_model.to(device)

with open("datasets/test.json", "r") as f:
    test_data = json.load(f)

# select only the first 1000 samples
test_data = test_data[:1000]

# Create datasets and dataloader
test_dataset = CustomDataset(test_data, max_len)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

loaded_model.eval()
correct_predictions = 0
total_predictions = 0

# Inference loop with tqdm
with torch.no_grad(), tqdm(total=len(test_dataloader), desc='Inference') as progress_bar:
    for batch in test_dataloader:
        masked_number = batch['masked_number'].to(device)
        actual_number = batch['actual_number'].to(device)

        # Forward pass
        predicted_number = loaded_model(masked_number.unsqueeze(1))

        if round(predicted_number.item()) == actual_number.item():
            correct_predictions += 1
        total_predictions += 1

        progress_bar.update(1)

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f'Accuracy: {accuracy * 100:.2f}%')