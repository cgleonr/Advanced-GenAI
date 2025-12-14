# Job Ad Analysis Pipeline

This project implements an end-to-end pipeline for analyzing job advertisements, including data preparation, Zone Classification (BERT), and Skill Extraction (LLM-based).

## Project Structure

- `data/`: Contains raw and processed data.
- `models/`: Stores trained BERT models.
- `output/`: Generated reports, plots, and extracted skills.
- `scripts/`: Python scripts for each stage of the pipeline.
- `configs/`: Configuration files (if any).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file with your Gemini API key:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

## Usage

### 1. Data Preparation
Preprocess the annotated data and align labels for BERT.
```bash
python scripts/preprocess_data.py
```

### 2. Model Training (Zone Classifier)
Train the BERT model to identify zones like "Fähigkeiten und Inhalte".
```bash
python scripts/train_bert.py
```
> **Note**: If no GPU is detected, the script defaults to a "Demo Mode" using only 5% of the data for 3 epochs. To force full training on CPU (very slow), modify `scripts/train_bert.py`.

### 3. Evaluation
Evaluate the trained model and generate a classification report and confusion matrix.
```bash
python scripts/evaluate_bert.py
```

### 4. Skill Extraction
Extract specific skills from the identified zones using Gemini.
```bash
python scripts/extract_skills.py
```
Output will be saved to `outputs/skills.json`.

## Pipeline Details

- **Model**: `bert-base-multilingual-cased` fine-tuned for token classification.
- **Skill Extraction**: Hybrid approach using BERT for zone identification and Gemini Pro for structured extraction.
- **Metrics**: Standard NER metrics (Precision, Recall, F1) via `seqeval`.
