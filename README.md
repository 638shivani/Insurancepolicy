# 🧠 PolicyMind Pro – AI-Powered Insurance Policy Analyzer

## 📌 Project Overview
PolicyMind Pro is an AI-powered Insurance Policy Analyzer that helps users understand complex insurance policy documents quickly and easily.

Users can upload an insurance policy PDF and ask questions in natural language. The system analyzes the document using Retrieval-Augmented Generation (RAG) and Google Gemini API to provide relevant answers based on the uploaded policy.

## 🎯 Objectives
- Simplify complex insurance policy documents.
- Provide quick answers to insurance-related queries.
- Retrieve relevant policy clauses using semantic search.
- Generate structured claim analysis results.
- Support multiple languages for better accessibility.

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Frontend:** Streamlit
- **PDF Extraction:** PyPDF2
- **Text Chunking:** LangChain
- **Embedding Model:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS
- **Generative AI:** Google Gemini 1.5 Flash
- **Data Processing:** NumPy

## 🔄 Project Workflow

1. **PDF Upload:** Users upload their insurance policy document in PDF format.
2. **Text Extraction:** PyPDF2 extracts text from the uploaded PDF.
3. **Text Chunking:** LangChain splits lengthy policy documents into smaller chunks of 500 characters with 100 characters of overlap.
4. **Embedding Generation:** Sentence Transformer converts text chunks into numerical embeddings.
5. **Vector Storage:** FAISS stores embeddings for efficient similarity-based retrieval.
6. **Query Processing:** The user's question is converted into an embedding.
7. **Relevant Information Retrieval:** FAISS retrieves the top 3 relevant policy chunks.
8. **AI Analysis:** Retrieved context and the user's question are passed to Google Gemini API through prompt-based context injection.
9. **Structured Output:** The system generates a JSON response containing:
   - Decision (Approved/Rejected)
   - Reason
   - Confidence Score
   - Medical Procedure
10. **Result Display:** Streamlit displays the analysis results through an interactive web interface.

## ✨ Key Features

- Upload and analyze insurance policy PDF documents.
- Ask questions using natural language.
- RAG-based document retrieval.
- Semantic search using FAISS.
- AI-powered policy analysis using Google Gemini.
- Structured JSON output for easy understanding.
- Interactive Streamlit web application.
- Multilingual support: English, Hindi, and Kannada.

## 🏗️ Architecture

**PDF Upload → Text Extraction → Text Chunking → Embeddings → FAISS Vector Database → Query Retrieval → Gemini API → Structured JSON Response → Streamlit UI**

## 📂 Project Structure

```text
PolicyMind-Pro/
│
├── app.py
├── requirements.txt
├── README.md
└── sample_policy.pdf
```

## 🚀 How to Run the Project

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project directory**

```bash
cd PolicyMind-Pro
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your Gemini API key**

Set your Google Gemini API key securely using Streamlit secrets or environment variables.

**5. Run the application**

```bash
streamlit run app.py
```

## 🔮 Future Enhancements

- Support for additional insurance document formats.
- Improved claim analysis and document comparison.
- Enhanced accuracy through advanced retrieval techniques.
- Deployment with secure user authentication.

## 👩‍💻 Developed By
**Shivani Ishwar Metri**

AI / Machine Learning Project
