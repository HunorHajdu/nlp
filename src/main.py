import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import json
from tqdm import tqdm

# Sample data
with open("datasets/train.json", "r") as f:
    data = json.load(f)

# select only the first 1000 samples
data = data[:1000]

should_train = False

# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item['masked'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        masked_number = torch.tensor(float(item['number']))  # Masked number as input
        actual_number = torch.tensor(float(item['number']))  # Actual number as target
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'masked_number': masked_number,
            'actual_number': actual_number
        }

# Create dataset and dataloader
dataset = CustomDataset(data, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.MSELoss()
if should_train:
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        total_loss = 0.0

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            masked_number = batch['masked_number'].to(device)
            actual_number = batch['actual_number'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_number = outputs.logits.squeeze(dim=-1)  # Adjust if necessary

            # Calculate the loss using Mean Squared Error
            loss = criterion(predicted_number, actual_number)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            progress_bar.set_postfix({'Loss': total_loss / (len(progress_bar) + 1e-12)})

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')


    # Save the trained model
    model.save_pretrained('masked_number_model')

# Inference
# Load the saved model
loaded_model = BertForSequenceClassification.from_pretrained('masked_number_model')
loaded_model.to(device) 

with open("datasets/test.json", "r") as f:
    test_data = json.load(f)

# select only the first 1000 samples
test_data = test_data[:1000]

# Create dataset and dataloader
test_dataset = CustomDataset(test_data, tokenizer, max_len)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

loaded_model.eval()
correct_predictions = 0
total_predictions = 0

# Inference loop with tqdm
with torch.no_grad(), tqdm(total=len(test_dataloader), desc='Inference') as progress_bar:
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        masked_number = batch['masked_number'].to(device)
        actual_number = batch['actual_number'].to(device)

        # Forward pass
        outputs = loaded_model(input_ids, attention_mask=attention_mask)
        predicted_number = torch.argmax(outputs.logits).item()

        if predicted_number == actual_number:
            correct_predictions += 1
        total_predictions += 1

        progress_bar.update(1)

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f'Accuracy: {accuracy * 100:.2f}%')