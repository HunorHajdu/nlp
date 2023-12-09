from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer
from datasets import load_dataset
import json

# Load the datasets
dataset = load_dataset('json', data_files={'train': 'datasets/train.json', 'dev': 'datasets/dev.json', 'test': 'datasets/test.json'})

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the new preprocess function
def preprocess_data(entry):
    # Extracting the relevant information from each entry
    question = entry['question']
    options = [entry['Option1'], entry['Option2']]
    answer = entry['answer']
    question_type = entry['type']
    
    # Converting the answer to a numerical label (0 or 1)
    answer_label = options.index(answer) if answer in options else -1
    
    # Tokenizing the question
    tokenized_input = tokenizer(question, padding='max_length', truncation=True)
    
    # Adding the answer label to the tokenized input
    tokenized_input['label'] = answer_label
    
    return tokenized_input

# Apply the new preprocess function to the datasets
tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['dev']
)

# Train the model
trainer.train()