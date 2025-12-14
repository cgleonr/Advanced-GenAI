"""
Two-Stage Skill Extraction Pipeline for Job Advertisements

This module implements a two-stage pipeline for extracting skills from job advertisements:
1. Zone Identification: Uses trained BERT model to identify "Fähigkeiten und Inhalte" zones
2. Skill Extraction: Uses LLM (Gemini or Ollama fallback) to extract structured skills

Features:
    - BERT-based zone identification with BIO tagging
    - LLM-based skill extraction with structured output
    - Automatic fallback from Gemini API to local Ollama
    - Skill phrase length validation (2-5 words)
    - Deduplication of extracted skills

LLM Fallback Strategy:
    1. Attempts to use Gemini API (if API key available)
    2. Falls back to local Ollama if Gemini fails
    3. Continues processing with whichever backend works

Output:
    - skills.json: Extracted skills per document with text snippets
"""

import os
import json
import torch
import numpy as np
import requests
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

# Pipeline configuration constants
MODEL_DIR = 'models/bert_zone_classifier'
INPUT_FILE = 'data/annotated.json'
OUTPUT_FILE = 'outputs/skills.json'
MAX_LEN = 512  # Maximum sequence length for BERT
OVERLAP = 128  # Overlap between sliding windows
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ollama configuration (fallback LLM)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def setup_gemini():
    if not API_KEY:
        return None
    genai.configure(api_key=API_KEY)
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model
    except:
        try:
            model = genai.GenerativeModel('gemini-pro')
            return model
        except:
            print("WARNING: Could not initialize Gemini model")
            return None

def extract_skills_with_ollama(text_segment):
    """Fallback: Extract skills using Ollama local LLM"""
    prompt = f"""You are an expert HR analyst. Extract specific skills from this job advertisement text.

Input Text: "{text_segment}"

Task: Extract a list of technical and soft skills mentioned in the text.
Constraints:
1. Each skill must be a phrase of 2 to 5 words.
2. Output ONLY a valid Python list of strings, nothing else.
3. Do not include generic phrases like "team player" unless specific context is given.
4. If no skills are found, return an empty list [].

Output Format (ONLY the list, no explanation): ["skill one", "skill two", ...]"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text_out = result.get('response', '').strip()
            
            # Clean up markdown
            for marker in ["```python", "```json", "```"]:
                text_out = text_out.replace(marker, "")
            
            # Extract list
            text_out = text_out.strip()
            if '[' in text_out and ']' in text_out:
                start = text_out.find('[')
                end = text_out.rfind(']') + 1
                text_out = text_out[start:end]
                
            try:
                skills = eval(text_out)
                if isinstance(skills, list):
                    return [s for s in skills if isinstance(s, str) and 2 <= len(s.split()) <= 5]
            except:
                return []
        return []
    except:
        return []

def extract_skills_with_llm(llm, text_segment, use_ollama_fallback=True):
    """Extract skills - tries Gemini first, falls back to Ollama if it fails"""
    
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
    
    # Try Gemini first
    if llm:
        try:
            response = llm.generate_content(prompt)
            text_out = response.text.strip()
            
            # Clean up markdown
            if text_out.startswith("```python"):
                text_out = text_out.replace("```python", "").replace("```", "")
            elif text_out.startswith("```json"):
                text_out = text_out.replace("```json", "").replace("```", "")
                
            # Parse list
            try:
                skills = eval(text_out)
                if isinstance(skills, list):
                    valid_skills = [s for s in skills if isinstance(s, str) and 2 <= len(s.split()) <= 5]
                    return valid_skills
            except:
                pass  # Fall through to Ollama fallback
        except Exception as e:
            print(f"Gemini error: {str(e)[:50]}... falling back to Ollama")
    
    # Fallback to Ollama
    if use_ollama_fallback:
        return extract_skills_with_ollama(text_segment)
    
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
            
            # Get label - FIX: use integer key
            label_id = int(preds[idx])
            # Try both int and str keys for compatibility
            label = id2label.get(label_id, id2label.get(str(label_id), 'O'))
            
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
    
    if llm:
        print(f"Using Gemini API (with Ollama fallback if needed)")
    else:
        print(f"Gemini not available - using Ollama fallback")
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
