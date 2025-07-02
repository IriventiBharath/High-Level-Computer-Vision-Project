# DocVQA: A Comparative Study

A comprehensive comparison of three different approaches for Document Visual Question Answering (DocVQA) evaluated on 300 document samples from the DocVQA dataset.

## 🎯 Results Summary

| Method | Exact Match (%) | F1 Score (%) | Approach |
|--------|----------------|--------------|----------|
| **VRDU OCR + LLM** | **49.3** | **55.3** | VRDU OCR extraction + Groq LLM |
| **Tesseract OCR + LLM** | **51.3** | **58.6** | Tesseract OCR + Groq LLM |
| **LLaVA-Llama3 (Direct)** | 9.67 | 15.75 | Direct vision-language model |

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
- ✅ Fastest processing time (~12.9 docs/min)
- ✅ Simple and reliable pipeline

**VRDU OCR + LLM (Competitive Performance)**
- ✅ Strong exact match rate (49.3%) - good precision
- ✅ Good F1 score (55.3%) - solid overall performance
- ✅ Advanced document understanding with layout awareness
- ⚠️ Slightly lower performance than Tesseract approach

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
- **LLM**: Groq Llama-3.1-8B-instant
- **Advantage**: Advanced document layout understanding

#### 2. Tesseract OCR + LLM Pipeline
```
Document Image → Tesseract OCR → Groq LLM → Answer
```
- **OCR Engine**: Tesseract
- **LLM**: Groq Llama-3.1-8B-instant
- **Advantage**: Fast and reliable text extraction

#### 3. LLaVA-Llama3 Direct Pipeline
```
Document Image → LLaVA-Llama3 Vision Model → Answer
```
- **Model**: LLaVA-Llama3 8B via Ollama
- **Advantage**: End-to-end vision-language understanding

## 📁 Project Structure

```
HLCV_project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (excluded from git)
├── .gitignore                        # Git ignore file
│
├── notebooks/
│   ├── LVLM_llava-llama3_8b.ipynb    # LLaVA-Llama3 implementation
│   ├── OCR_Tesseract.ipynb           # Tesseract OCR + LLM implementation
│   └── OCR_VRDU_QA_copy.ipynb        # VRDU OCR + LLM implementation
│
├── results/
│   ├── vlm_results.csv               # LLaVA-Llama3 results
│   ├── OCR_results_tesseract.csv     # Tesseract results
│   └── OCR_VRDU_results.csv          # VRDU results
│
└── docvqa_samples_300/
    ├── images/                       # Document images (300 samples)
    │   ├── doc_0000.png
    │   ├── doc_0001.png
    │   └── ...
    └── metadata.json                 # Questions and ground truth answers
```

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

## 🔧 Technical Implementation Details

### OCR Processing
- **VRDU**: Uses Docling framework with SmolDocling transformer for document understanding
- **Tesseract**: Traditional OCR with pytesseract Python wrapper
- **Text Preprocessing**: Normalization, punctuation removal, whitespace handling

### LLM Integration
- **Model**: Groq Llama-3.1-8B-instant for OCR approaches
- **Temperature**: 0 for deterministic responses
- **Prompt Strategy**: Structured prompts requesting concise answers
- **Error Handling**: Graceful handling of API failures and timeouts

### Evaluation Pipeline
- **Preprocessing**: Text normalization for fair comparison
- **Exact Match**: Strict string matching after normalization
- **F1 Score**: Token-level precision and recall calculation
- **Real-time Results**: CSV output with detailed per-sample results

## 📈 Analysis & Insights

### Why OCR-First Approaches Outperform Direct Vision Models

1. **Text Recognition Specialization**: OCR engines are specifically designed for text extraction
2. **Document Layout Understanding**: VRDU models understand document structure better
3. **LLM Strengths**: LLMs excel at reasoning over extracted text
4. **Pipeline Optimization**: Two-stage approach allows optimization of each component

### Performance Trade-offs

- **VRDU vs Tesseract**: VRDU provides better contextual understanding but Tesseract is more precise
- **Speed vs Accuracy**: Tesseract is fastest but VRDU provides highest quality
- **Resource Requirements**: Direct vision models require significant GPU resources

### Recommendations

1. **For Production Systems**: Use VRDU OCR + LLM for best overall performance
2. **For High-Speed Processing**: Use Tesseract OCR + LLM for optimal speed/accuracy balance
3. **For Simple Documents**: Tesseract may be sufficient for straightforward text extraction
4. **Avoid Direct Vision Models**: Current vision-language models are not optimized for document QA

## 🛠️ Requirements

### Python Packages
```
langchain-groq>=0.3.5
langchain-docling>=0.1.0
langchain-core>=0.3.67
langchain-ollama>=0.2.0
python-dotenv>=1.1.1
matplotlib>=3.10.0
pillow>=11.2.1
tqdm>=4.67.1
numpy>=2.0.2
pytesseract>=0.3.13
pandas>=2.0.0
```

### System Requirements
- **GPU**: Recommended for VRDU and LLaVA approaches
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: ~2GB for models and data
- **API Keys**: Groq API key for LLM access

### External Dependencies
- **Tesseract OCR**: System-level installation required
- **Ollama**: Required for LLaVA-Llama3 approach
- **CUDA**: Optional but recommended for GPU acceleration

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
- Experiment with larger vision-language models (GPT-4V, Gemini Pro Vision)
- Investigate hybrid approaches combining OCR and vision models
- Evaluate on different document types (forms, tables, handwritten text)
- Optimize prompt engineering for vision models

## 📄 License

This project is for educational and research purposes. Please ensure compliance with the terms of service of all APIs and models used.

---

*This comparative study was conducted to evaluate different approaches for document visual question answering and provides insights into the current state of document AI technologies.*
