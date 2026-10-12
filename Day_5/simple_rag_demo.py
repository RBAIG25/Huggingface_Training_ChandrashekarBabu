import torch
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ==========================================
# Step 1: Prepare Your Private Knowledge Base
# ==========================================
# In a real app, this text would be extracted from PDFs or Markdown files
documents = [
    "The 2026 Space Odyssey Mission is scheduled to launch on October 14, 2026, targeting Jupiter's moon Europa.",
    "The crew of the 2026 Space Odyssey consists of Commander Elena Rostova and Dr. Marcus Vance.",
    "Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf.",
    "Retrieval-Augmented Generation (RAG) optimizes LLM outputs by querying a reliable external knowledge base."
]

# ==========================================
# Step 2: Create Vector Embeddings
# ==========================================
print("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert texts to dense vector arrays
doc_embeddings = embedding_model.encode(documents)
embedding_dimension = doc_embeddings.shape[1]

# Initialize and populate the FAISS Index
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(doc_embeddings).astype("float32"))
print(f"Indexed {index.ntotal} document chunks.")

# ==========================================
# Step 3: Retrieval Function
# ==========================================
def retrieve(query, k=1):
    """Finds the most semantically relevant text chunk for a query."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k)
    
    # Extract the closest matching document text
    retrieved_chunk = documents[indices[0][0]]
    return retrieved_chunk

# ==========================================
# Step 4: Generation Pipeline (RAG)
# ==========================================
print("Loading Generator LLM (Gemma3)...")
model_id = "google/gemma-3-1b-it"

# Initialize text generation pipeline
generator = pipeline(
    "text-generation", 
    model=model_id, 
    dtype=torch.bfloat16, 
    device_map="auto"
)

def ask_rag(question):
    """Executes the complete RAG loop: Retrieve -> Prompt -> Generate."""
    # 1. Fetch relevant context
    context = retrieve(question, k=1)
    
    # 2. Structure prompt with instructions and context
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question using ONLY the provided context.",
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        },
    ]
    
    # 3. Format the chat prompt structure
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 4. Generate the final output
    outputs = generator(prompt, do_sample=True, temperature=0.6)
    response = outputs[0]["generated_text"][len(prompt):]
    return response

# ==========================================
# Step 5: Test the System
# ==========================================
query = "When is the Space Odyssey mission launching and who is on the crew?"
print(f"\n--- Testing RAG ---\nQuery: {query}")

answer = ask_rag(query)
print(f"\nAnswer:\n{answer.strip()}")

