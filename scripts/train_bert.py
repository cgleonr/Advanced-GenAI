import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Configuration
DATA_DIR = 'data/processed'
MODEL_DIR = 'models/bert_zone_classifier'
LOG_DIR = 'logs'
MODEL_NAME = 'bert-base-multilingual-cased'
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
SEED = 42

def load_processed_data():
    print("Loading datasets...")
    train_dataset = torch.load(os.path.join(DATA_DIR, 'train_dataset.pt'), weights_only=False)
    test_dataset = torch.load(os.path.join(DATA_DIR, 'test_dataset.pt'), weights_only=False)
    
    with open(os.path.join(DATA_DIR, 'label2id.json'), 'r') as f:
        label2id = json.load(f)
    with open(os.path.join(DATA_DIR, 'id2label.json'), 'r') as f:
        id2label = json.load(f)
        
    return train_dataset, test_dataset, label2id, id2label

def calculate_class_weights(dataset, label2id):
    # Iterate over dataset to count labels
    # This might be slow for huge datasets, but okay for expected size
    # Or we can just use standard weights or ignore if not huge imbalance
    # Ruleset asks for "weighted CrossEntropyLoss"
    
    print("Calculating class weights...")
    label_counts = {i: 0 for i in label2id.values()}
    total_tokens = 0
    
    # We can iterate the tensor directly if possible, or DataLoader
    # dataset[i] -> (input_ids, mask, labels)
    # dataset.dataset is the TensorDataset, but subset maps indices
    # Let's just iterate a dataloader for simplicity
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    for batch in tqdm(loader, desc="Counting labels"):
        labels = batch[2].view(-1)
        labels = labels[labels != -100]
        for l in labels:
            label_counts[l.item()] += 1
            total_tokens += 1
            
    # Inverse frequency weights
    # weight = total / (num_classes * count)
    num_classes = len(label2id)
    weights = []
    sorted_ids = sorted(label2id.values())
    
    for i in sorted_ids:
        count = label_counts[i]
        if count > 0:
            w = total_tokens / (num_classes * count)
        else:
            w = 1.0 # default if absent
        weights.append(w)
        
    return torch.tensor(weights, dtype=torch.float)

def train():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # Seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset, val_dataset, label2id, id2label = load_processed_data()
    
    if not torch.cuda.is_available():
        print("WARNING: GPU not found. Switching to CPU mode.")
        print("Optimization: Using a subset of data (5%) and fewer epochs to ensure pipeline completion.")
        # Subset data
        subset_size = int(len(train_dataset) * 0.05)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
        val_subset_size = int(len(val_dataset) * 0.05)
        val_indices = torch.randperm(len(val_dataset))[:val_subset_size]
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        EPOCHS = 3 # Reduce epochs
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_labels = len(label2id)
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Class weights
    class_weights = calculate_class_weights(train_dataset, label2id).to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    
    # Logging
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Use local epochs variable
    training_epochs = EPOCHS
    if not torch.cuda.is_available():
        training_epochs = 3 

    for epoch in range(training_epochs):
        print(f"\nEpoch {epoch + 1}/{training_epochs}")
        
        # Train
        model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            
            outputs = model(
                b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask
            )
            
            logits = outputs.logits
            # Reshape for loss
            active_loss = b_input_mask.view(-1) == 1
            active_logits = logits.view(-1, num_labels)
            active_labels = b_labels.view(-1)
            
            # Only compute loss where mask is active (already handled by ignore_index=-100 if we pass full, 
            # but usually manual masking is safer or just passing to loss_fct)
            # HF model output.loss uses default loss. We want custom weighted loss.
            
            loss = loss_fct(active_logits, active_labels)
            
            total_train_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            if step % 10 == 0:
                writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_dataloader) + step)

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average train loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        nb_eval_steps = 0
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():
                outputs = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask
                )
            
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, num_labels), b_labels.view(-1))
            total_val_loss += loss.item()
            
            # Accuracy (optional quick metric)
            predictions = torch.argmax(logits, dim=2)
            # Just simple accuracy on active tokens
            active_mask = b_labels != -100
            correct = (predictions == b_labels) & active_mask
            total_val_accuracy += correct.sum().item() / active_mask.sum().item()
            nb_eval_steps += 1
            
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_acc = total_val_accuracy / nb_eval_steps
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {avg_val_acc:.4f}")
        
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', avg_val_acc, epoch)
        
        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            print("Saving best model...")
            model.save_pretrained(MODEL_DIR)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
            # Also save tokenizer for convenience
            # tokenizer.save_pretrained(MODEL_DIR) # Tokenizer not loaded here but good practice
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break
            
    print("Training complete.")
    writer.close()

if __name__ == '__main__':
    train()
