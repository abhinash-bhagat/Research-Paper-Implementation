from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader

# Load the Magicoder-OSS-Instruct-75K dataset
dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")

# Load the PyTorch version of GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the tokenizer's pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA Settings
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # Corrected typo here
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with LoRA
lora_model = get_peft_model(model, lora_config)

# Tokenize the dataset
# Preprocessing: For this task, we'll use the 'problem' as input and 'solution' as output
def preprocess_function(examples):
    inputs = examples['problem']
    targets = examples['solution']
    
    # Tokenize both problem and solution
    tokenized_inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt", padding="max_length")
    tokenized_targets = tokenizer(targets, max_length=512, truncation=True, return_tensors="pt", padding="max_length")
    
    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze(),  # Remove extra dimensions
        "labels": tokenized_targets["input_ids"].squeeze()     # Remove extra dimensions
    }

# Apply the preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create a PyTorch DataLoader for batching
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)

# Define optimizer
optimizer = AdamW(lora_model.parameters(), lr=5e-5)

# Training loop
lora_model.train()  # Set the model to training mode
for epoch in range(3):  # Run for a few epochs
    for batch in train_dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = lora_model(input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Loss: {loss.item()}")

# Save the fine-tuned LoRA model
lora_model.save_pretrained("fine-tuned-code-model-lora")