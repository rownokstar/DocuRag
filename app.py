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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant # আমরা Qdrant ব্যবহার করছি
from qdrant_client import QdrantClient


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
# (I-3: gemini-embedding-001, Qdrant)

# --- +++ এই ফাংশনটি 503 এরর ঠিক করার জন্য আপডেট করা হয়েছে +++ ---
@st.cache_resource(ttl=3600)
def get_embeddings_model(_api_key: str):
    """Google Generative AI এমবেডিং মডেল ক্যাশ করে।"""
    try:
        # --- +++ এইখানে মূল সমাধান +++ ---
        # আমরা GoogleGenerativeAIEmbeddings-কে সরাসরি API Key দিচ্ছি
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=_api_key  # কী-টি এখানে পাস করতে হবে
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {e}")
        st.error(f"Error initializing embeddings. Is your API key correct? Error: {e}")
        return None

# --- এই ফাংশনটি 504 এরর ঠিক করার জন্য ব্যাচিং সহ আপডেট করা আছে ---
def create_vector_store(all_chunks: List[Document], embeddings_model: Any) -> Qdrant:
    """চাঙ্ক এবং এমবেডিং ব্যবহার করে একটি Qdrant ভেক্টর স্টোর তৈরি করে। (ব্যাচ প্রসেসিং সহ)"""
    if not all_chunks:
        logger.warning("No chunks to process. Returning empty vector store.")
        return None
        
    logger.info(f"Creating Qdrant index from {len(all_chunks)} chunks...")
    
    # ইন-মেমোরি Qdrant ক্লায়েন্ট শুরু করুন
    client = QdrantClient(location=":memory:")
    collection_name = "docurag-collection"
    
    try:
        # ভেক্টর স্টোর অবজেক্টটি প্রথমে তৈরি করুন
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings_model
        )
        
        # এমবেডিং-এর জন্য ব্যাচ সাইজ (Batch Size) নির্ধারণ করুন
        batch_size = 100 # একবারে ১০০টি চাঙ্ক প্রসেস করবে
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            logger.info(f"Adding batch {i//batch_size + 1}/{len(all_chunks)//batch_size + 1} with {len(batch)} chunks...")
            vector_store.add_documents(batch) # ব্যাচ যোগ করুন
            
        logger.info("Qdrant index created successfully with all batches.")
        return vector_store
        
    except Exception as e:
        # এটি প্রায়শই ভুল API কী-এর কারণে ঘটে (অথবা ব্যাচিং ফেইল হলে)
        logger.error(f"Qdrant index creation/batching failed: {e}")
        st.error(f"Failed to create vector store. Is your API key correct? Error: {e}")
        return None


# --- ৪. RAG পাইপলাইন এবং জেনারেশন ---
# (I-KA-4: RAG Pipeline, Gemini API, Citations)

def get_rag_response(query: str, vector_store: Any, genai_model: Any) -> Dict[str, Any]:
    """
    RAG পাইপলাইন এক্সিকিউট করে: রিট্রিভ, প্রম্পট তৈরি, এবং জেনারেট।
    """
    logger.info(f"Received query: {query}")
    
    # বিশেষ কমান্ড চেক (E-3 / E-4)
    if query.strip().lower().startswith("show table"):
        parts = query.split()
        if len(parts) >= 3:
            table_key = parts[2]
            if 'tables' in st.session_state and table_key in st.session_state.tables:
                return {"answer": f"Displaying table: {table_key}", "table_data": st.session_state.tables[table_key]}
            else:
                return {"answer": f"Sorry, table '{table_key}' not found in processed documents."}
        else:
             return {"answer": "Please specify the table key. You can find keys in the 'Processed Tables' section."}

    # ১. রিট্রিভ (Retrieve)
    retriever = vector_store.as_retriever(search_k=4) # (I-4: top-k (3-5))
    retrieved_docs = retriever.get_relevant_documents(query)
    
    if not retrieved_docs:
        logger.warning("No relevant documents found.")
        return {"answer": "I'm sorry, I couldn't find any relevant information in the uploaded documents to answer your question."}

    # ২. কনটেক্সট এবং প্রম্পট তৈরি (Augment)
    context = ""
    sources = set()
    for doc in retrieved_docs:
        context += f"\n--- Context from {doc.metadata['source']}, Page {doc.metadata['page']} ---\n"
        context += doc.page_content
        context += f"\n--- End of Context ---\n"
        sources.add(f"{doc.metadata['source']}, Page {doc.metadata['page']}")
        
    citation_str = " (Source: " + ", ".join(sorted(list(sources))) + ")"
    
    system_prompt = f"""
    You are DocuRAG, a specialized AI assistant for document analysis.
    Your task is to answer the user's question based *only* on the provided context.
    Do not use any external knowledge.
    If the answer is not found in the context, state that clearly.
    Be concise and professional.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    ANSWER (based *only* on the context):
    """
    
    # ৩. জেনারেট (Generate)
    logger.info("Sending prompt to Gemini-Pro...")
    try:
        response = genai_model.generate_content(system_prompt)
        final_answer = response.text + citation_str # (S: Output: Each answer must include citation)
        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return {"answer": f"Error generating response from AI model. Please check logs. Error: {e}"}


# --- ৫. চ্যাটবট ইন্টারফেস (Streamlit UI) ---
# (I-5: Streamlit UI)

def main():
    st.set_page_config(page_title="DocuRAG", page_icon="📄", layout="wide")
    st.title("📄 DocuRAG: AI Document Intelligence PoC")
    
    # --- সাইডবার: API কী এবং ফাইল আপলোড ---
    with st.sidebar:
        st.header("Configuration")
        
        # Google API কী ইনপুট
        api_key = st.text_input("Enter your Google API Key", type="password")
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                st.session_state.api_key_configured = True
                logger.info("Google API Key configured.")
            except Exception as e:
                st.error(f"Failed to configure API key: {e}")
                st.session_state.api_key_configured = False
        else:
            st.session_state.api_key_configured = False
            st.warning("Please enter your Google API Key to proceed.")

        st.divider()
        
        # ফাইল আপলোডার
        uploaded_files = st.file_uploader(
            "Upload your PDF documents",
            type="pdf",
            accept_multiple_files=True,
            disabled=not st.session_state.api_key_configured
        )
        
        # --- প্রসেস বাটন (এই লজিকটি AttributeError-এর জন্য ঠিক করা আছে) ---
        if st.button("Process Documents", disabled=not uploaded_files or not st.session_state.api_key_configured):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a few minutes for large files or OCR."):
                    all_chunks = []
                    
                    # এমবেডিং মডেল লোড করা
                    # এইখানে api_key পাস করা হচ্ছে
                    embeddings_model = get_embeddings_model(api_key)
                    
                    if embeddings_model:
                        for file in uploaded_files:
                            st.info(f"Ingesting '{file.name}'...")
                            # ১. ইনজেশন (টেক্সট, ওসিআর, টেবিল)
                            page_data = process_uploaded_pdf(file)
                            # ২. প্রি-প্রসেসিং এবং চাঙ্কিং
                            chunks = chunk_data(file.name, page_data)
                            all_chunks.extend(chunks)
                        
                        # ৩. এমবেডিং এবং ভেক্টর স্টোর তৈরি (সঠিক লজিক সহ)
                        if all_chunks:
                            # প্রথমে একটি অস্থায়ী ভেরিয়েবলে স্টোর করুন
                            vector_store = create_vector_store(all_chunks, embeddings_model)
                            
                            if vector_store:
                                # সফল হলেই সেশন স্টেটে সেভ করুন
                                st.session_state.vector_store = vector_store
                                st.session_state.documents_processed = True
                                st.success(f"Processed {len(uploaded_files)} documents ({len(all_chunks)} chunks). Ready to chat!")
                            else:
                                # যদি ভেক্টর স্টোর তৈরি না হয় (যেমন: ভুল API কী বা 503/504 এরর)
                                st.error("Failed to create vector store. Please check your API Key or console logs.")
                                st.session_state.documents_processed = False 
                        else:
                            st.warning("No text or table data could be extracted from the documents.")
                            st.session_state.documents_processed = False 
                    else:
                        st.error("Cannot process documents: Embeddings model failed to load.")
                        st.session_state.documents_processed = False 

        st.divider()
        
        # প্রসেস করা টেবিলের তালিকা (E-3)
        if 'tables' in st.session_state and st.session_state.tables:
            with st.expander("View Processed Tables"):
                st.info("Use 'show table <key>' to display a table in the chat.")
                for key in st.session_state.tables.keys():
                    st.code(key)

    # --- মেইন চ্যাট ইন্টারফেস ---
    
    # সেশন স্টেট ইনিশিয়ালাইজেশন
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to DocuRAG! Please upload your documents and I'll help you analyze them."}]
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
        
    if "tables" not in st.session_state:
        st.session_state.tables = {}

    # চ্যাট হিস্ট্রি প্রদর্শন
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "table_data" in message:
                st.dataframe(message["table_data"]) # (E-3: Display table)
                # (E-4: Chart Visualization)
                try:
                    # চার্টিং এর জন্য একটি সিম্পল চেষ্টা
                    df_numeric = message["table_data"].apply(pd.to_numeric, errors='coerce').dropna(axis=1)
                    if not df_numeric.empty and len(df_numeric.columns) > 1:
                        st.bar_chart(df_numeric, x=df_numeric.columns[0], y=df_numeric.columns[1])
                except Exception:
                    pass # চার্ট করা না গেলে সাইলেন্টলি ফেইল হবে

    # ইউজার ইনপুট হ্যান্ডলিং
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.api_key_configured:
            st.info("Please configure your Google API Key in the sidebar first.")
        elif not st.session_state.documents_processed:
            st.info("Please upload and process your documents first.") 
        else:
            # ইউজার মেসেজ প্রদর্শন
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # অ্যাসিস্ট্যান্ট রেসপন্স জেনারেট
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    genai_model = genai.GenerativeModel('gemini-pro')
                    response_dict = get_rag_response(
                        prompt,
                        st.session_state.vector_store,
                        genai_model
                    )
                    
                    st.markdown(response_dict["answer"])
                    
                    # রেসপন্স সেশন স্টেটে যোগ করা
                    assistant_message = {"role": "assistant", "content": response_dict["answer"]}
                    if "table_data" in response_dict:
                        assistant_message["table_data"] = response_dict["table_data"]
                        st.dataframe(response_dict["table_data"])
                    
                    st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
