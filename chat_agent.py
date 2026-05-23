import os
import cohere
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
cohere_key = os.getenv("COHERE_API_KEY")
if not cohere_key:
    raise ValueError("🚨 COHERE_API_KEY not found in .env file!")

# Initialize Cohere Client
co = cohere.Client(cohere_key)

def generate_rag_response(question):
    print("\n" + "="*60)
    print(f"👤 USER QUESTION: {question}")
    print("="*60)
    
    # ------------------------------------------
    # 1. RETRIEVAL (The "R")
    # ------------------------------------------
    print("🔍 Searching Milvus for relevant insurance clauses...")
    
    # Generate embedding for the question
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode([question]).tolist()[0]

    # Connect to Milvus
    db_client = MilvusClient(uri="tcp://127.0.0.1:19530")
    
    # Retrieve top 3 relevant chunks
    search_res = db_client.search(
        collection_name="insurance_policies",
        data=[question_embedding],
        limit=3,
        output_fields=["text"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )

    # Format the retrieved context
    context_text = ""
    for hit in search_res[0]:
        context_text += f"\n---\n{hit['entity']['text']}\n"

    # ------------------------------------------
    # 2. GENERATION (The "G")
    # ------------------------------------------
    print("🧠 Command-R is synthesizing an answer...\n")
    
    # Cohere's 'preamble' acts as the System Message
    preamble = """You are an AI Insurance Assistant. Your goal is to provide 
    accurate answers based ONLY on the provided policy documents. 
    If the answer isn't in the context, say 'I cannot find that information in the documents.'"""

    # Generate response
    response = co.chat(
        model='command-r-08-2024', # Optimized specifically for RAG tasks
        message=f"CONTEXT FROM POLICIES:\n{context_text}\n\nUSER QUESTION: {question}",
        preamble=preamble,
        temperature=0.3
    )

    # ------------------------------------------
    # 3. FINAL OUTPUT
    # ------------------------------------------
    print("🤖 COHERE AGENT RESPONSE:")
    print("-" * 60)
    print(response.text)
    print("-" * 60 + "\n")

if __name__ == "__main__":
    # Test with a specific insurance query
    query = "What are the key features and eligibility criteria of the PM-POSHAN scheme?"
    generate_rag_response(query)