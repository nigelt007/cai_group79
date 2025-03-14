import streamlit as st
import numpy as np
import re
import tempfile
from datetime import datetime
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize classifier once for input guardrail
classifier = pipeline("zero-shot-classification", 
                    model="typeform/distilbert-base-uncased-mnli")

# Streamlit UI Configuration
st.set_page_config(page_title="Multi-File Financial Analyzer", layout="wide")
st.title("📊 Comparative Financial Analysis System")

# Sidebar Controls
with st.sidebar:
    st.header("Configuration Panel")
    model_choice = st.selectbox("LLM Model", 
                              ["deepseek-r1:1.5b", "llama3.2:1b"],
                              help="Choose the core analysis engine")
    chunk_size = st.slider("Document Chunk Size", 500, 2000, 1000)
    rerank_threshold = st.slider("Re-ranking Threshold", 0.0, 1.0, 0.5)

# File Upload Handling for multiple files
uploaded_files = st.file_uploader("Upload 2 Financial PDFs", 
                                type="pdf", 
                                accept_multiple_files=True)

if len(uploaded_files) == 2:
    all_docs = []
    with st.spinner("Processing Multiple Financial Documents..."):
        for uploaded_file in uploaded_files:
            # Create temporary file for each PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load and process each document
            loader = PDFPlumberLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)

        # Combined Document Processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", "\. ", "! ", "? ", " ", ""]
        )
        documents = text_splitter.split_documents(all_docs)

        # Hybrid Retrieval Setup for combined documents
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embedder)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6]
        )

        # Re-ranking Model
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Financial Analysis LLM Configuration
        llm = Ollama(model=model_choice)
        PROMPT_TEMPLATE = """
        As a senior financial analyst, analyze the following context from multiple financial reports:
        1. Compare key metrics between both documents
        2. Identify trends across reporting periods
        3. Highlight significant differences or similarities
        4. Provide integrated risk assessment
        5. Offer comprehensive recommendations
        
        Context: {context}
        Question: {question}
        
        Format with clear section headers and bullet points.
        Maintain comparative analysis throughout.
        Keep under 300 words.
        """
        qa_prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        llm_chain = LLMChain(llm=llm, prompt=qa_prompt)  # Proper LLMChain initialization

    # Interactive Q&A Interface
    st.header("🔍 Cross-Document Financial Inquiry")
    
    # Suggested Comparative Questions
    comparative_questions = [
        "Compare revenue growth between both fiscal years",
        "Analyze changes in debt structure across both reports",
        "Show expense ratio differences between the two years",
        "What are the main liquidity changes across both periods?",
        "How does net profit margin compare between the two reports?"
    ]
    user_query = st.selectbox("Sample Comparative Questions", 
                            [""] + comparative_questions)
    user_input = st.text_input("Or enter custom comparative query:", 
                             value=user_query)

    if user_input:
        # Input Validation Guardrail
        classification = classifier(user_input, 
                                  ["financial comparison", "other"],
                                  multi_label=False)
        if classification['scores'][0] < 0.2:
            st.error("Query not comparative/financial. Ask about financial comparisons between documents.")
            st.stop()

        with st.spinner("Performing Cross-Document Analysis..."):
            # Hybrid Document Retrieval
            initial_docs = ensemble_retriever.get_relevant_documents(user_input)
            
            # Context Re-ranking
            doc_pairs = [(user_input, doc.page_content) for doc in initial_docs]
            rerank_scores = cross_encoder.predict(doc_pairs)
            sorted_indices = np.argsort(rerank_scores)[::-1]
            ranked_docs = [initial_docs[i] for i in sorted_indices]
            filtered_docs = [d for d, s in zip(ranked_docs, rerank_scores) 
                           if s > rerank_threshold][:7]
            
            # Confidence Calculation
            confidence_score = np.mean(rerank_scores[sorted_indices][:3]) * 100
            confidence_score = min(100, max(0, round(confidence_score, 1)))

            # Response Generation
            context = "\n".join([doc.page_content for doc in filtered_docs])
            analysis = llm_chain.run(
                context=context, 
                question=user_input
            )
            
            # Response Cleaning
            clean_analysis = re.sub(r"<think>|</think>|\n{3,}", "", analysis)
            clean_analysis = re.sub(r'(\d)([A-Za-z])', r'\1 \2', clean_analysis)
            clean_analysis = re.sub(r'(\d{1,3})(\d{3})', r'\1,\2', clean_analysis)

            # Results Display
            st.subheader("Integrated Financial Analysis")
            st.markdown(f"```\n{clean_analysis}\n```")
            st.progress(int(confidence_score)/100)
            st.caption(f"Analysis Confidence: {confidence_score}%")

            # Export Functionality
            if st.button("Generate Comparative Report"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_content = f"COMPARATIVE QUERY: {user_input}\n\nANALYSIS:\n{clean_analysis}"
                st.download_button("Download Full Report", export_content,
                                 file_name=f"Comparative_Analysis_{timestamp}.txt",
                                 mime="text/plain")

elif len(uploaded_files) > 0:
    st.warning("Please upload exactly 2 financial documents for comparative analysis")
else:
    st.info("Please upload 2 PDF financial reports to begin comparative analysis")