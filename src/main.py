import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import json
from tqdm import tqdm

# Sample data
with open("datasets/train.json", "r") as f:
    data_train = json.load(f)

with open("datasets/test.json", "r") as f:
    data_test = json.load(f)

with open("datasets/dev.json", "r") as f:
    data_val = json.load(f)

sample_size = None  # Number of samples to use for training, validation, and test, set to None to use all samples

# Limit the number of samples to sample_size
data_train = data_train[:sample_size]
data_test = data_test[:sample_size]
data_val = data_val[:sample_size]

should_train = False

# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 512

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        masked_sentence = item['masked'].replace('[Num]', self.tokenizer.mask_token)
        inputs = self.tokenizer(
            masked_sentence,
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

# Create datasets and dataloaders for training, validation, and test
train_dataset = CustomDataset(data_train, tokenizer, max_len)
val_dataset = CustomDataset(data_val, tokenizer, max_len)
test_dataset = CustomDataset(data_test, tokenizer, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Define the optimizer
num_epochs = 3
validation_interval = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Learning rate scheduler

criterion = torch.nn.MSELoss()

if should_train:
    for epoch in range(num_epochs):
        # Training
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} (Training)', leave=False)
        total_loss = 0.0

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            masked_number = batch['masked_number'].to(device)
            actual_number = batch['actual_number'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_number = outputs.logits.squeeze(dim=-1)

            # Calculate the loss using Mean Squared Error
            loss = criterion(predicted_number, actual_number)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            progress_bar.set_postfix({'Loss': total_loss / (len(progress_bar) + 1e-12)})

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}')
        # validate the model every validation_interval epochs
        if (epoch + 1) % validation_interval == 0:
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)', leave=False):
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_masked_number = val_batch['masked_number'].to(device)
                    val_actual_number = val_batch['actual_number'].to(device)

                    val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
                    val_predicted_number = val_outputs.logits.squeeze(dim=-1)

                    val_loss += criterion(val_predicted_number, val_actual_number).item()

            average_val_loss = val_loss / len(val_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss:.4f}')

        # Adjust learning rate
        scheduler.step()

# Inference
# Load the saved model
loaded_model = BertForSequenceClassification.from_pretrained('masked_number_model')
loaded_model.to(device) 

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
