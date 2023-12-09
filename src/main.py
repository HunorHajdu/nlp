from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer
from datasets import load_dataset

# Load the datasets
dataset = load_dataset('json', data_files={'train': 'datasets/train.json', 'dev': 'datasets/dev.json', 'test': 'datasets/test.json'})

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the new preprocess function
def preprocess_data(examples):
    # Tokenizing the questions
    tokenized_inputs = tokenizer(examples['question'], padding='max_length', truncation=True)
    
    # Converting the answers to numerical labels (0 or 1)
    answer_labels = [options.index(answer) if answer in options else -1 for answer, options in zip(examples['answer'], zip(examples['Option1'], examples['Option2']))]
    
    # Adding the answer labels to the tokenized inputs
    tokenized_inputs['labels'] = answer_labels
    
    return tokenized_inputs

# Apply the new preprocess function to the datasets
tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Define the model with the correct number of labels
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
