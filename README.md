# 🏥 Medical RAG with Qwen 2.5 & Hybrid Search

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-Qwen%202.5-violet)

## 📖 Introduction
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer medical questions using the **PubMedQA** dataset.

It combines **Hybrid Retrieval** (Semantic + Keyword search) with a **Fine-tuned Qwen 2.5 (0.5B)** model to achieve high accuracy in medical reasoning. The system also employs an **Ensemble Soft Voting** strategy to further improve performance.

## 🚀 Key Features
* **Hybrid Retrieval Engine:** Combines **ChromaDB** (Dense Vector Search) and **BM25** (Sparse Keyword Search) to retrieve the most relevant medical contexts.
* **Efficient Fine-tuning:** Uses **QLoRA** (4-bit Quantization) to fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on consumer hardware.
* **RAG Pipeline:** Integrates retrieved context into the prompt to reduce hallucinations.
* **Ensemble Learning:** Implements Soft Voting between **BioLinkBERT** and the Fine-tuned **Qwen** model for robust decision-making.

## 📂 Project Structure
The project is modularized for scalability and ease of maintenance:

```text
medical-rag-project/
├── data/                   # Dataset and raw inputs
├── notebooks/              # Jupyter Notebooks for execution
│   ├── 01_generate_embeddings.ipynb  # Step 1: Build Vector DB
│   ├── 02_finetune_qwen.ipynb        # Step 2: Train QLoRA Adapter
│   └── 03_rag_evaluation.ipynb       # Step 3: RAG & Ensemble Eval
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py           # Configuration & Paths
│   ├── data_utils.py       # Data formatting logic
│   ├── vector_store.py     # ChromaDB construction
│   ├── retrieval.py        # Hybrid Search implementation
│   └── model_utils.py      # Model loading & merging
├── requirements.txt        # Dependencies
└── README.md
```
