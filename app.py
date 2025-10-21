import os
import io
import logging
import tempfile
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import camelot
import pytesseract
from PIL import Image
from typing import List, Tuple, Dict, Any

import google.generativeai as genai
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings # আমরা লোকাল এমবেডিং ব্যবহার করছি
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant # আমরা Qdrant ব্যবহার করছি
# from qdrant_client import QdrantClient # ব্যাচিং-এর জন্য এটি আর দরকার নেই


# --- সিস্টেম কনফিগারেশন এবং লগিং ---

# লগার সেটআপ (P: Clear logs for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tesseract OCR-এর পাথ (যদি এটি সিস্টেম PATH-এ না থাকে)
# প্রয়োজন হলে এটি আনকমেন্ট করুন এবং আপনার Tesseract-এর পাথ দিন
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- ১. PDF ইনজেশন এবং ডেটা এক্সট্রাকশন ---
# (I-1: PyMuPDF, Tesseract, Camelot)

def extract_text_with_ocr(page: fitz.Page, file_name: str, page_num: int) -> str:
    """একটি পৃষ্ঠা থেকে টেক্সট এবং OCR ইমেজ বের করে।"""
    text = page.get_text("text")
    image_text = ""
    
    # স্ক্যান করা পৃষ্ঠার জন্য OCR চালানো (P: Handle missing or scanned text)
    try:
        if len(text.strip()) < 100:  # যদি পৃষ্ঠায় খুব কম টেক্সট থাকে, তবে OCR চেষ্টা করুন
            images = page.get_images(full=True)
            if images:
                logger.info(f"Running OCR on {file_name} page {page_num}...")
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Tesseract দিয়ে OCR
                    image_text += pytesseract.image_to_string(pil_image)
                    logger.info(f"OCR extracted {len(image_text)} chars from image {img_index} on page {page_num}.")
    except Exception as e:
        logger.warning(f"Could not run OCR on {file_name} page {page_num}: {e}")

    return text + "\n" + image_text

def extract_tables_with_camelot(temp_pdf_path: str, file_name: str, page_num: int) -> str:
    """Camelot ব্যবহার করে একটি নির্দিষ্ট পৃষ্ঠা থেকে টেবিল বের করে।"""
    table_content = ""
    try:
        # Camelot শুধুমাত্র ফাইল পাথ থেকে পড়তে পারে, বাইট স্ট্রীম থেকে নয়
        tables = camelot.read_pdf(temp_pdf_path, pages=str(page_num), flavor='lattice')
        
        if tables.n > 0:
            logger.info(f"Extracted {tables.n} tables from {file_name} page {page_num}.")
            for i, table in enumerate(tables):
                table_df = table.df
                table_content += f"\n--- Table {i+1} on Page {page_num} ---\n"
                table_content += table_df.to_markdown(index=False)
                table_content += f"\n--- End of Table {i+1} ---\n"
                
                # টেবিল ডেটা সেশন স্টেটে সংরক্ষণ (E-3: Table Extraction)
                if 'tables' not in st.session_state:
                    st.session_state.tables = {}
                table_key = f"{file_name}_p{page_num}_t{i+1}"
                st.session_state.tables[table_key] = table_df
                
    except Exception as e:
        # (P: Table extraction failure fallback)
        logger.warning(f"Camelot failed to extract tables from {file_name} page {page_num}: {e}")
        
    return table_content

def process_uploaded_pdf(uploaded_file: Any) -> List[Tuple[int, str, str]]:
    """
    একটি আপলোড করা PDF ফাইল প্রসেস করে।
    PyMuPDF, Tesseract, এবং Camelot ব্যবহার করে টেক্সট, ইমেজ (OCR), এবং টেবিল বের করে।
    """
    file_name = uploaded_file.name
    logger.info(f"Starting processing for: {file_name}")
    
    # Camelot-এর জন্য একটি অস্থায়ী ফাইলে PDF সংরক্ষণ করা
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_pdf_path = temp_file.name

    all_page_data = []
    
    try:
        # PyMuPDF (fitz) দিয়ে PDF খোলা
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")

        for page_num_zero_based in range(len(doc)):
            page_num_one_based = page_num_zero_based + 1
            page = doc.load_page(page_num_zero_based)
            
            # ১. টেক্সট এবং OCR এক্সট্রাকশন
            page_text = extract_text_with_ocr(page, file_name, page_num_one_based)
            if page_text.strip():
                all_page_data.append((page_num_one_based, page_text, "text"))
            
            # ২. টেবিল এক্সট্রাকশন (Camelot)
            table_text = extract_tables_with_camelot(temp_pdf_path, file_name, page_num_one_based)
            if table_text.strip():
                all_page_data.append((page_num_one_based, table_text, "table"))

        doc.close()
        
    except Exception as e:
        logger.error(f"Failed to process PDF {file_name}: {e}")
        st.error(f"Error processing {file_name}. Is it a valid PDF?")
    finally:
        # অস্থায়ী ফাইল মুছে ফেলা
        os.remove(temp_pdf_path)
        logger.info(f"Finished processing for: {file_name}")

    return all_page_data


# --- ২. ডেটা প্রি-প্রসেসিং এবং চাঙ্কিং ---
# (I-2: Chunking, Cleaning, Metadata)

def chunk_data(file_name: str, processed_data: List[Tuple[int, str, str]]) -> List[Document]:
    """প্রসেস করা ডেটাকে চাঙ্ক করে এবং মেটাডেটা সহ LangChain Document অবজেক্ট তৈরি করে।"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # (I-2: ~500-800 tokens)
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    all_chunks = []
    for page_num, content, content_type in processed_data:
        # (I-2: Clean and normalize)
        cleaned_content = ' '.join(content.split())
        
        # মেটাডেটা তৈরি
        metadata = {
            "source": file_name,
            "page": page_num,
            "type": content_type
        }
        
        # টেক্সট স্প্লিট করা
        chunks = text_splitter.split_text(cleaned_content)
        
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=metadata))
            
    logger.info(f"Created {len(all_chunks)} chunks for {file_name}.")
    return all_chunks


# --- ৩. এমবেডিং এবং ভেক্টরাইজেশন ---
# (I-3: Local Embeddings, Qdrant)

@st.cache_resource(ttl=3600)
def get_embeddings_model():
    """ফ্রি, ওপেন-সোর্স HuggingFace এমবেডিং মডেল লোড করে।"""
    try:
        logger.info("Loading local embeddings model (all-MiniLM-L6-v2)...")
        # এই মডেলটি Streamlit-এর CPU-তে চলবে
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("Local embeddings model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize local embeddings model: {e}")
        st.error(f"Error initializing local embeddings model. Error: {e}")
        return None

# --- এই ফাংশনটি "Collection not found" এরর ঠিক করার জন্য সিম্পল করা হয়েছে ---
def create_vector_store(all_chunks: List[Document], embeddings_model: Any) -> Qdrant:
    """চাঙ্ক এবং এমবেডিং ব্যবহার করে একটি Qdrant ভেক্টর স্টোর তৈরি করে।"""
    if not all_chunks:
        logger.warning("No chunks to process. Returning empty vector store.")
        return None
        
    logger.info(f"Creating Qdrant index from {len(all_chunks)} chunks...")
    
    try:
        # লোকাল এমবেডিং মডেলের জন্য ব্যাচিং-এর দরকার নেই
        # from_documents ফাংশনটি কালেকশন তৈরি করে এবং ডেটা যোগ করে, একবারে।
        vector_store = Qdrant.from_documents(
            all_chunks,
            embeddings_model,
            location=":memory:",  # ইন-মেমোরি স্টোর ব্যবহার করুন
            collection_name="docurag-collection"
        )
            
        logger.info("Qdrant index created successfully.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Qdrant index creation failed: {e}")
        st.error(f"Failed to create vector store. Error: {e}")
        return None


# --- ৪. RAG পাইপলাইন এবং জেনারেশন ---
# (I-KA-4: RAG Pipeline, Gemini API, Citations)

def get_rag_response(query: str, vector_store: Any, genai_model:
