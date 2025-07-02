# DocVQA: A Comparative Study

A comprehensive comparison of three different approaches for Document Visual Question Answering (DocVQA) evaluated on 300 document samples from the DocVQA dataset.

## 🎯 Results Summary

| Method | Exact Match (%) | F1 Score (%) | Approach |
|--------|----------------|--------------|----------|
| **Tesseract OCR + LLM** | **51.3** | **58.6** 
| **VRDU OCR + LLM** | **49.3** | **55.3** | VRDU OCR + Llama-3.1-8B-instant via Groq 
| **LLaVA-Llama3 (Direct)** | 9.67 | 15.75 | Direct vision-language model 

### Key Findings
- **OCR-first approaches significantly outperform direct vision-language models** for document QA tasks
- **Tesseract OCR + LLM achieves the best performance overall** with 51.3% exact match and 58.6% F1 score
- **VRDU OCR + LLM performs competitively** with 49.3% exact match and 55.3% F1 score
- **LLaVA-Llama3 direct approach struggles** with complex document understanding, achieving only 9.67% exact match

## 📊 Performance Analysis

### Strengths and Weaknesses

**Tesseract OCR + LLM (Best Overall)**
- ✅ Highest exact match rate (51.3%) - most precise answers
- ✅ Highest F1 score (58.6%) - excellent balance of precision and recall
- ✅ Fastest processing time (25 min for 300 documents)
- ✅ Simple and reliable pipeline

**VRDU OCR + LLM (Competitive Performance)**
- ✅ Strong exact match rate (49.3%) - good precision
- ✅ Good F1 score (55.3%) - solid overall performance
- ✅ Advanced document understanding with layout awareness
- ⚠️ Slower processing time (4+ hours for 300 documents)

**LLaVA-Llama3 Direct (Poorest Performance)**
- ❌ Lowest performance across all metrics
- ❌ Struggles with fine-grained document text recognition
- ❌ Not optimized for document-specific tasks
- ✅ No OCR preprocessing required

## 🔍 Methodology

### Dataset
- **Source**: DocVQA dataset samples
- **Size**: 300 document images
- **Format**: PNG images with corresponding questions and ground truth answers
- **Task**: Answer questions based on document content

### Evaluation Metrics
- **Exact Match (EM)**: Percentage of predictions that exactly match ground truth
- **F1 Score**: Token-level F1 score measuring partial matches and answer quality

### Approaches Tested

#### 1. VRDU OCR + LLM Pipeline
```
Document Image → VRDU OCR (Docling) → Groq LLM → Answer
```
- **OCR Engine**: Docling with SmolDocling transformer model
- **LLM**: Llama-3.1-8B-instant via Groq API
- **Advantage**: Advanced document layout understanding

#### 2. Tesseract OCR + LLM Pipeline
```
Document Image → Tesseract OCR → Groq LLM → Answer
```
- **OCR Engine**: Tesseract
- **LLM**: Llama-3.1-8B-instant via Groq API
- **Advantage**: Fast and reliable text extraction

#### 3. LLaVA-Llama3 Direct Pipeline
```
Document Image → LLaVA-Llama3 Vision Model → Answer
```
- **Model**: LLaVA-Llama3 8B via Ollama
- **Advantage**: End-to-end vision-language understanding

## 💻 Compute Infrastructure

- **VRDU OCR**: Google Colab T4 GPU → RTX 3060 local GPU (due to compute limitations)
- **Processing Time**: Tesseract (25 min) vs VRDU (4+ hours) for 300 documents  
- **LLM**: Llama-3.1-8B-instant via Groq API for all OCR approaches

## 🚀 Usage Instructions

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# For LLaVA-Llama3 approach
# Install Ollama and pull the model
ollama pull llava-llama3

# For Tesseract approach
# Install Tesseract OCR
sudo apt install tesseract-ocr  # Ubuntu/Debian
# or
brew install tesseract  # macOS
```

### Environment Setup
Create a `.env` file with your API keys:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Running the Experiments

1. **VRDU OCR + LLM**:
   ```bash
   jupyter notebook OCR_VRDU_QA_copy.ipynb
   ```

2. **Tesseract OCR + LLM**:
   ```bash
   jupyter notebook OCR_Tesseract.ipynb
   ```

3. **LLaVA-Llama3 Direct**:
   ```bash
   jupyter notebook LVLM_llava-llama3_8b.ipynb
   ```

## 🔧 Technical Details

- **LLM**: Llama-3.1-8B-instant via Groq API (temperature=0)
- **OCR**: VRDU (Docling framework) vs Tesseract
- **Evaluation**: Exact Match + F1 Score with text normalization

## 📈 Key Insights

**Why OCR-First Approaches Outperform Direct Vision Models:**
- OCR engines are specialized for text extraction
- LLMs excel at reasoning over extracted text  
- Two-stage pipeline allows component optimization

**Recommendations:**
- **High-Speed Processing**: Use Tesseract OCR + LLM
- **Best Quality**: Use VRDU OCR + LLM (when compute allows)
- **Avoid**: Direct vision models for document QA

## ⚠️ Limitations

### Dataset Constraints
- **Small evaluation set**: Only 300 samples from DocVQA dataset
- **Single domain**: Limited to document QA, no generalization testing
- **No data splits**: All samples used for evaluation (no train/test separation)

### Technical Limitations
- **OCR Text Truncation**: Tesseract approach truncates content to 1500 characters
- **Processing Speed**: VRDU requires 4+ hours vs Tesseract's 25 minutes
- **Memory Requirements**: VRDU approach demands significant GPU memory
- **API Dependencies**: Reliance on Groq API availability and rate limits
- **Hardware Sensitivity**: Performance varies with GPU/CPU configuration

### Methodology Constraints  
- **Limited Metrics**: Only exact match and F1 score evaluated
- **No Statistical Testing**: Missing confidence intervals or significance tests
- **Basic Preprocessing**: Simple text normalization, no advanced techniques
- **Single Model Variants**: One model tested per approach category

### Implementation Issues
- **No Error Recovery**: Failed samples are skipped without retry mechanisms
- **Sequential Processing**: No parallel processing implementation
- **No Reproducibility Controls**: Missing random seed and version pinning
- **Basic Error Handling**: Generic exception catching without detailed logging

## 📊 Detailed Results

The complete results for all 300 samples are available in the CSV files:
- `vlm_results.csv`: LLaVA-Llama3 predictions and scores
- `OCR_results_tesseract.csv`: Tesseract OCR + LLM results
- `OCR_VRDU_results.csv`: VRDU OCR + LLM results

Each file contains:
- Document ID and filename
- Original question
- Ground truth answer(s)
- Predicted answer
- Exact match score (0/1)
- F1 score (0.0-1.0)
- OCR content (where applicable)

## 🎓 Academic Context

This study demonstrates the effectiveness of traditional OCR + LLM pipelines compared to modern vision-language models for document question answering tasks. The results suggest that specialized document processing pipelines remain superior to general-purpose vision models for text-heavy document understanding tasks.

### Future Work
- **Non-OCR vision model comparisons** (coming soon)
- Experiment with larger vision-language models (GPT-4V, Gemini Pro Vision)
- Investigate hybrid approaches combining OCR and vision models
- Evaluate on different document types (forms, tables, handwritten text)

