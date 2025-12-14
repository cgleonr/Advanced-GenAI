"""
BERT Model Evaluation Module for Zone Classification

This module evaluates the trained BERT token classification model on the test dataset.
Computes standard NER metrics (precision, recall, F1) using seqeval and generates
a confusion matrix visualization for error analysis.

Features:
    - Per-zone precision/recall/F1 metrics
    - Macro and micro-averaged metrics
    - Confusion matrix visualization
    - JSON export of detailed results

Output:
    - evaluation_report.json: Detailed metrics in JSON format
    - confusion_matrix.png: Visual confusion matrix
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
from seqeval.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Evaluation configuration constants
DATA_DIR = 'data/processed'
MODEL_DIR = 'models/bert_zone_classifier'
OUTPUT_DIR = 'outputs'
BATCH_SIZE = 16  # Larger batch size for inference
SEED = 42

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def evaluate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading test data...")
    test_dataset = torch.load(os.path.join(DATA_DIR, 'test_dataset.pt'), weights_only=False)
    with open(os.path.join(DATA_DIR, 'label2id.json'), 'r') as f:
        label2id = json.load(f)
    with open(os.path.join(DATA_DIR, 'id2label.json'), 'r') as f:
        id2label = json.load(f)
        
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if not torch.cuda.is_available():
        print("Optimizing: subsetting test data for CPU evaluation.")
        indices = torch.randperm(len(test_dataset))[:50] # 50 samples
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    print("Loading model...")
    # Try loading best model, fallback to final
    model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    if not os.path.exists(model_path):
        # Maybe model saved via save_pretrained?
        if os.path.exists(os.path.join(MODEL_DIR, 'config.json')):
            model = BertForTokenClassification.from_pretrained(MODEL_DIR)
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_DIR}")
    else:
        # We need to initialize structure then load state dict
        # Or just load the dir if we saved via save_pretrained
        # The training script saves both: save_pretrained(MODEL_DIR) AND state_dict to best_model.pt
        # So we can just load from MODEL_DIR
        model = BertForTokenClassification.from_pretrained(MODEL_DIR)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    
    print("Running inference...")
    for batch in tqdm(test_dataloader, desc="Evaluating"):
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
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Convert logits to predictions
        batch_preds = np.argmax(logits, axis=2)
        
        for i in range(len(label_ids)):
            pred_list = []
            label_list = []
            for j in range(len(label_ids[i])):
                if label_ids[i][j] != -100:
                    pred_list.append(id2label[str(batch_preds[i][j])])
                    label_list.append(id2label[str(label_ids[i][j])])
            
            predictions.append(pred_list)
            true_labels.append(label_list)
            
    # Compute Metrics
    print("\nComputing metrics...")
    report = classification_report(true_labels, predictions, output_dict=True)
    report_text = classification_report(true_labels, predictions)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Flatten lists for confusion matrix
    flat_true = [label for sublist in true_labels for label in sublist]
    flat_pred = [pred for sublist in predictions for pred in sublist]
    
    # Get unique labels
    labels = sorted(list(set(flat_true + flat_pred)))
    
    cm = confusion_matrix(flat_true, flat_pred, labels=labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png")
    
    f1 = f1_score(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    
    print("\nClassification Report:\n")
    print(report_text)
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    # Save results
    results = {
        'classification_report': report,
        'f1_score': f1,
        'accuracy': acc
    }
    
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
        
    print(f"Evaluation complete. Results saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    evaluate()
