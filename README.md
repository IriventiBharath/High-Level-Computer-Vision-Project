# DocVQA: A Comparative Study

A comprehensive comparison of four different approaches for Document Visual Question Answering (DocVQA) evaluated on both the DocVQA dataset (300 samples) and a custom daily tasks dataset (10 samples).

### ## All findings should be interpreted as exploratory and indicative, not definitive. Further evaluation on larger, more diverse datasets is necessary to confirm generalizability. ## ###

## 🎯 Results Summary

### DocVQA Dataset (300 samples)

| Method | Exact Match (%) | F1 Score (%) | Approach |
|--------|----------------|--------------|----------|
| **Donut DocVQA Fine-Tuned** | **55.0** | **62.9** | End-to-end vision transformer |
| **Tesseract OCR + LLM** | 51.3 | 58.6 | Traditional OCR + Llama-3.1-8B |
| **VRDU OCR + LLM** | 49.3 | 55.3 | VRDU OCR + Llama-3.1-8B-instant via Groq |
| **LLaVA-Llama3 (Direct)** | 9.67 | 15.75 | Direct vision-language model |

### Custom Daily Tasks Dataset (10 samples)

| Method | Exact Match (%) | F1 Score (%) | Approach |
|--------|----------------|--------------|----------|
| **Tesseract OCR + LLM** | **57.1** | **57.1** | Traditional OCR + Llama-3.1-8B |
| **VRDU OCR + LLM** | 45.1 | 28.3 | VRDU OCR + Llama-3.1-8B-instant via Groq |
| **LLaVA-Llama3 (Direct)** | 20.0 | 29.2 | Direct vision-language model |
| **Donut DocVQA Fine-Tuned** | 10.0 | 10.0 | End-to-end vision transformer |

### Key Findings
- **Donut fine-tuned model achieves best performance on DocVQA dataset** with 55.0% exact match and 62.9% F1 score
- **Tesseract OCR + LLM shows strong generalization** across both datasets
- **VRDU OCR + LLM performs well on standard DocVQA** but struggles with custom daily task images
- **LLaVA-Llama3 direct approach consistently underperforms** across all document types
- **Domain adaptation is crucial** - Donut excels on DocVQA but fails on out-of-domain daily tasks

## 📊 Performance Analysis

### Strengths and Weaknesses

**Donut DocVQA Fine-Tuned (Best on DocVQA)**
- ✅ Highest performance on DocVQA dataset (55.0% EM, 62.9% F1)
- ✅ End-to-end vision transformer approach
- ✅ No OCR preprocessing required
- ❌ Poor generalization to out-of-domain tasks (10.0% EM on daily tasks)
- ❌ Requires domain-specific fine-tuning

**Tesseract OCR + LLM (Most Generalizable)**
- ✅ Consistent performance across both datasets
- ✅ Strong exact match rates (51.3% DocVQA, 57.1% daily tasks)
- ✅ Fastest processing time (25 min for 300 documents)
- ✅ Simple and reliable pipeline
- ✅ Best generalization capability

**VRDU OCR + LLM (Layout-Aware)**
- ✅ Strong performance on DocVQA (49.3% EM, 55.3% F1)
- ✅ Advanced document understanding with layout awareness
- ❌ Inconsistent performance on daily tasks (45.1% EM, 28.3% F1)
- ⚠️ Slower processing time (4+ hours for 300 documents)

**LLaVA-Llama3 Direct (Poorest Performance)**
- ❌ Lowest performance across all metrics and datasets
- ❌ Struggles with fine-grained document text recognition
- ❌ Not optimized for document-specific tasks
- ✅ No OCR preprocessing required

## 🔍 Methodology

### Datasets
- **DocVQA Dataset**: 300 document images from the DocVQA dataset
- **Custom Daily Tasks Dataset**: 10 document images from daily activities (receipts, forms, etc.)
- **Format**: PNG images with corresponding questions and ground truth answers
- **Task**: Answer questions based on document content

### Evaluation Metrics
- **Exact Match (EM)**: Percentage of predictions that exactly match ground truth
- **F1 Score**: Token-level F1 score measuring partial matches and answer quality

### Approaches Tested

#### 1. Donut DocVQA Fine-Tuned
```
Document Image → Donut Vision Transformer → Answer
```
- **Model**: Donut fine-tuned on DocVQA dataset
- **Architecture**: End-to-end vision transformer
- **Advantage**: Specialized for document understanding tasks

#### 2. VRDU OCR + LLM Pipeline
```
Document Image → VRDU OCR (Docling) → Groq LLM → Answer
```
- **OCR Engine**: Docling with SmolDocling transformer model
- **LLM**: Llama-3.1-8B-instant via Groq API
- **Advantage**: Advanced document layout understanding

#### 3. Tesseract OCR + LLM Pipeline
```
Document Image → Tesseract OCR → Groq LLM → Answer
```
- **OCR Engine**: Tesseract
- **LLM**: Llama-3.1-8B-instant via Groq API
- **Advantage**: Fast and reliable text extraction

#### 4. LLaVA-Llama3 Direct Pipeline
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

1. **Donut DocVQA Fine-Tuned**:
   ```bash
   jupyter notebook OCR_Donut.ipynb
   ```

2. **VRDU OCR + LLM**:
   ```bash
   jupyter notebook OCR_VRDU_QA.ipynb
   ```

3. **Tesseract OCR + LLM**:
   ```bash
   jupyter notebook OCR_Tesseract.ipynb
   ```

4. **LLaVA-Llama3 Direct**:
   ```bash
   jupyter notebook LVLM_llava-llama3_8b.ipynb
   ```

## 🔧 Technical Details

- **LLM**: Llama-3.1-8B-instant via Groq API (temperature=0)
- **OCR**: VRDU (Docling framework) vs Tesseract
- **Evaluation**: Exact Match + F1 Score with text normalization

## 📈 Key Insights

**Performance Hierarchy:**
1. **Domain-specific models excel on matching tasks**: Donut fine-tuned on DocVQA achieves best DocVQA performance
2. **OCR + LLM approaches show strong generalization**: Tesseract maintains consistent performance across domains
3. **Layout-aware OCR has mixed results**: VRDU excels on structured documents but struggles with daily tasks
4. **Direct vision models consistently underperform**: LLaVA struggles across all document types

**Domain Adaptation is Critical:**
- Fine-tuned models (Donut) excel on similar data but fail on out-of-domain tasks
- Traditional OCR + LLM approaches show better generalization
- Real-world deployment requires robust cross-domain performance

**Recommendations:**
- **For DocVQA-like tasks**: Use Donut fine-tuned model
- **For general document QA**: Use Tesseract OCR + LLM (best speed/performance balance)
- **For structured documents**: Consider VRDU OCR + LLM (when compute allows)
- **Avoid**: Direct vision models for document QA tasks

## ⚠️ Limitations

### Dataset Constraints
- **Small evaluation set**: Only 300 samples from DocVQA dataset and 10 from personal dataset


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

The complete results are available in the CSV files:

### DocVQA Dataset Results (300 samples)
- `results/Donut_finetuned_results.csv`: Donut fine-tuned predictions and scores
- `results/vlm_results.csv`: LLaVA-Llama3 predictions and scores
- `results/OCR_tesseract_results.csv`: Tesseract OCR + LLM results
- `results/OCR_VRDU_results.csv`: VRDU OCR + LLM results

### Custom Daily Tasks Dataset Results (10 samples)
- `results_NEWDATA/OCR_DONUT_RESULTS_NEWDATASET.csv`: Donut results on daily tasks
- `results_NEWDATA/vlm_results_NEWDATASET.csv`: LLaVA-Llama3 results on daily tasks
- `results_NEWDATA/OCR_results_tesseract_NEWDATASET.csv`: Tesseract OCR + LLM results on daily tasks
- `results_NEWDATA/OCR_VRDU_RESULTS_NEWDATASET.csv`: VRDU OCR + LLM results on daily tasks

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

## Custom Daily Tasks Dataset Analysis

The custom daily tasks dataset (10 samples) consists of real-world document images including receipts, forms, and everyday documents. Key observations:

- **Domain shift significantly impacts performance**: Models trained/optimized for academic DocVQA data struggle with real-world daily documents
- **Tesseract OCR + LLM shows best robustness**: Maintains strong performance (57.1% EM) across domain boundaries
- **Donut fine-tuned model fails completely**: Drops from 55.0% to 10.0% EM, highlighting overfitting to DocVQA domain
- **VRDU inconsistent performance**: Good EM (45.1%) but poor F1 (28.3%), suggesting partial answer quality issues

This highlights the importance of domain generalization in real-world document understanding applications. 

