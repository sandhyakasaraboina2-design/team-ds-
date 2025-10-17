       
all necessary packages

# Import libraries
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import numpy as np

# Step 1: Upload PDFs in Colab
from google.colab import files

print("Upload your PDFs:")
uploaded_files = files.upload()

# Step 2: Extract and chunk text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        # Simple cleaning
        text = text.strip().replace('\n', ' ')
        full_text.append(text)
    return " ".join(full_text)

# Chunking text into ~500 tokens (approx 300-350 words)
def chunk_text(text, max_chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i+max_chunk_size])
        chunks.append(chunk)
    return chunks

# Extract and chunk all PDFs
all_chunks = []
source_refs = []  # to keep track of which chunk came from which file
for filename in uploaded_files.keys():
    print(f"Processing {filename} ...")
    text = extract_text_from_pdf(filename)
    chunks = chunk_text(text)
    all_chunks.extend(chunks)
    source_refs.extend([filename]*len(chunks))

print(f"Total chunks created: {len(all_chunks)}")

# Step 3: Create embeddings with SentenceTransformers
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)

print("Generating embeddings for text chunks...")
chunk_embeddings = embedder.encode(all_chunks, show_progress_bar=True)

# Step 4: Build FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

print(f"FAISS index built with {index.ntotal} vectors.")

# Step 5: Setup IBM Granite LLM model pipeline for answer generation
from transformers import pipeline

print("Loading IBM Granite model...")
pipe = pipeline("text-generation", model="ibm-granite/granite-3.2-2b-instruct")

# Step 6: Function to perform semantic search and generate answer
def ask_question(question, k=5):
    # Embed question
    question_emb = embedder.encode([question])
    # Search top k relevant chunks
    D, I = index.search(np.array(question_emb), k)
    relevant_chunks = [all_chunks[i] for i in I[0]]
    sources = [source_refs[i] for i in I[0]]

    # Prepare prompt for LLM: include context + question
    context = "\n\n".join([f"Source ({sources[i]}): {relevant_chunks[i]}" for i in range(k)])
    prompt = (
        f"Use the following context to answer the question accurately and concisely.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Generate answer
    response = pipe(prompt, max_new_tokens=256, do_sample=False, truncation=True)
    answer = response[0]['generated_text'][len(prompt):].strip()

    return answer, sources

# Step 7: Example interaction in Colab
while True:
    question = input("\nEnter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer, sources = ask_question(question)
    print("\nAnswer:", answer)
    print("Referenced Sources:", set(sources))