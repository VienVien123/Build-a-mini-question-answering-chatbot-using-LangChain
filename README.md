# Build a mini question-answering chatbot using LangChain

## Overview
This project leverages the `Vinallama-7B` model to build a question-answering system using LangChain. The system processes data from PDFs and stores the embeddings in a `FAISS` vector database, allowing for efficient semantic search.  You can also use data in various formats, and similarly, build the database in different ways. LangChain supports them all.

Learn more about LangChain at [LangChain Documentation](https://www.langchain.com/).

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
    - [2.1 Clone Repository](#21-clone-repository)
    - [2.2 Set Up Virtual Environment and Install Libraries](#22-set-up-virtual-environment-and-install-libraries)
    - [2.3 Download Models](#23-download-models)

---

## 1. Prerequisites
- **Python Version:** 3.8 or higher
- **Operating System:** Windows, macOS, or Linux

---

## 2. Installation
First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/vinallama-llm-train.git
cd vinallama-llm-train
```
### 2.1 Clone Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/vinallama-llm-train.git
cd vinallama-llm-train
```
### 2.2 Set Up Virtual Environment and Install Libraries

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
source .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```
### 2.3 Download Models
Create a `models` directory, navigate to it, and download the model files from the provided links:

```bash
mkdir models
cd models
```
[Hugging Face - MiniLM](https://huggingface.co/caliex/all-MiniLM-L6-v2-f16.gguf/tree/main) và
[Hugging Face - Vinallama](https://huggingface.co/vilm/vinallama-7b-chat-GGUF/tree/main)

After downloading, place the models in the `models` directory for easy access by the application.

## Additional Information
To customize the data input format, database storage options, and other settings, consult the [LangChain Documentation](https://www.langchain.com/) for extensive support.


