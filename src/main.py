from transformers  import  BertForSequenceClassification, Trainer, TrainingArguments
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
    
    # Create a copy of the input_ids
    labels = [ids.copy() for ids in tokenized_inputs['input_ids']]

    # Identify the position of the masked token (assuming it's already there)
    mask_positions = [[i for i, id in enumerate(ids) if id == tokenizer.mask_token_id] for ids in tokenized_inputs['input_ids']]

    # Set the labels for non-masked positions to -100
    for ids, mask_pos in zip(labels, mask_positions):
        ids[:] = [-100 if i not in mask_pos else id for i, id in enumerate(ids)]
    
    # Adding the labels to the tokenized inputs
    tokenized_inputs['labels'] = labels
    
    return tokenized_inputs

# Apply the new preprocess function to the datasets
tokenized_datasets = dataset.map(preprocess_data, batched=True)
mask_token = tokenizer.mask_token
# Define the model with the correct number of labels

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8192,
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
