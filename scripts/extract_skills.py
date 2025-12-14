import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertTokenizerFast, BertForTokenClassification
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

# Configuration
MODEL_DIR = 'models/bert_zone_classifier'
INPUT_FILE = 'data/annotated.json' # Or new data
OUTPUT_FILE = 'outputs/skills.json'
MAX_LEN = 512
OVERLAP = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_gemini():
    if not API_KEY:
        return None
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    return model

def extract_skills_with_llm(llm, text_segment):
    if not llm:
        return []
    
    prompt = f"""
    You are an expert HR analyst. Your task is to extract specific skills from a job advertisement text segment.
    
    Input Text: "{text_segment}"
    
    Task: Extract a list of technical and soft skills mentioned in the text.
    Constraints:
    1. Each skill must be a phrase of 2 to 5 words.
    2. Output must be a valid Python list of strings.
    3. Do not include generic phrases like "team player" unless specific context is given.
    4. If no skills are found, return an empty list [].
    
    Output Format: ["skill one", "skill two", ...]
    """
    
    try:
        response = llm.generate_content(prompt)
        text_out = response.text.strip()
        # Clean up markdown if present
        if text_out.startswith("```python"):
            text_out = text_out.replace("```python", "").replace("```", "")
        elif text_out.startswith("```json"):
            text_out = text_out.replace("```json", "").replace("```", "")
            
        # Parse list
        try:
            skills = eval(text_out)
            if isinstance(skills, list):
                # Filter length constraint
                valid_skills = [s for s in skills if isinstance(s, str) and 2 <= len(s.split()) <= 5]
                return valid_skills
        except:
            print(f"Failed to parse LLM output: {text_out}")
            return []
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return []
    return []

def get_zones(model, tokenizer, text, id2label, target_label='Fähigkeiten und Inhalte'):
    # Tokenize
    inputs = tokenizer(
        text,
        max_length=MAX_LEN,
        stride=OVERLAP,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).cpu().numpy()
    
    # Reconstruct zones
    # This is tricky with sliding windows. Simple approach: take first window coverage or merge.
    # For extraction, we just need text content. We can iterate windows and collect text where label matches.
    
    skill_text_segments = []
    
    # Simplified reconstruction: iterate all windows, extract text spans matching label
    # Deduplicate based on offsets?
    
    # We'll collect spans (start_char, end_char) then merge.
    spans = []
    
    offset_mapping = inputs['offset_mapping']
    
    for i in range(len(input_ids)):
        preds = predictions[i]
        offsets = offset_mapping[i]
        
        current_span_start = -1
        
        for idx, (start, end) in enumerate(offsets):
            if start == end: continue # special token
            
            # Get label
            label_id = preds[idx]
            label = id2label.get(str(label_id), 'O') # id2label keys might be strings in json
            
            # Check if matches target (careful with B- / I- prefixes)
            # Assuming label is like 'Fähigkeiten und Inhalte' directly or 'B-Fähigkeiten und Inhalte'
            # Let's normalize
            if target_label in label:
                if current_span_start == -1:
                    current_span_start = start
            else:
                if current_span_start != -1:
                    # End of span
                    spans.append((current_span_start, offsets[idx-1][1]))
                    current_span_start = -1
                    
        if current_span_start != -1:
             spans.append((current_span_start, offsets[-2][1])) # approximate end
             
    # Merge overlapping spans
    if not spans:
        return []
        
    spans.sort()
    merged = []
    if spans:
        curr_start, curr_end = spans[0]
        for next_start, next_end in spans[1:]:
            # If next start is within current end + small buffer (e.g. 5 chars)
            if next_start <= curr_end + 5: 
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))
        
    # Extract text
    segments = []
    for start, end in merged:
        # Verify length
        segment = text[start:end].strip()
        if len(segment) > 10: # Minimum length
             segments.append(segment)
             
    # Log found zones for debugging
    if segments:
        print(f"DEBUG: Found {len(segments)} skill zones.")
    
    return segments

def main():
    if not os.path.exists(MODEL_DIR): 
        print(f"Model directory {MODEL_DIR} not found.")
        return

    # Load Model (assuming trained)
    # Note: Training might still be running. Ideally run this after.
    try:
        model = BertForTokenClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
    except:
        print("Model not ready yet.")
        return
        
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    
    with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
        # id2label is in config usually if saved with save_pretrained
        # But we also have it in data/processed
        pass
    
    # We need id2label from data/processed to match logic if needed, 
    # but model.config.id2label should be correct if we initialized it right during training.
    id2label = model.config.id2label
    
    # Load Data (Simulate usage on raw text, but we use annotated.json for demo)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    items = data if isinstance(data, list) else [data]
    # Limit for demo
    items = items[:5] 
    
    llm = setup_gemini()
    results = []
    
    print(f"Extracting skills from {len(items)} documents...")
    
    for item in tqdm(items):
        text = item.get('content_clean') or item.get('data', {}).get('content_clean')
        if not text: continue
        
        # 1. Identify Zones
        segments = get_zones(model, tokenizer, text, id2label)
        
        # 2. Extract Skills from segments
        doc_skills = []
        for seg in segments:
            # Skip very short segments
            if len(seg) < 20: continue
            
            skills = extract_skills_with_llm(llm, seg)
            doc_skills.extend(skills)
            
        results.append({
            'text_snippet': text[:100],
            'extracted_skills': list(set(doc_skills))
        })
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Skills extracted to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
