import os
import sys

# Force PySpark to use the current virtual environment's Python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(binary_content):
    """Takes binary PDF data and returns extracted text."""
    try:
        import fitz
        pdf_document = fitz.open(stream=binary_content, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def chunk_text(text):
    """Splits the massive text into manageable, overlapping chunks."""
    if text.startswith("Error"): return []
    
    # We use overlapping chunks so clauses that cross paragraphs aren't lost
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def generate_embeddings(iterator):
    """Processes chunks and generates vector embeddings inside the Spark Workers."""
    from sentence_transformers import SentenceTransformer
    
    # Initialize the model ONLY once per worker partition to save RAM
    # all-MiniLM-L6-v2 is a blazing fast, highly accurate model for RAG
    print("Loading Embedding Model on Worker...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for file_path, chunks in iterator:
        if not chunks:
            continue
        
        # Generate embeddings for all chunks in this file simultaneously
        embeddings = model.encode(chunks).tolist()
        
        # Yield each chunk and its vector data
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{os.path.basename(file_path)}_chunk_{i}"
            # We are yielding a dictionary that is ready to be pushed to a Vector DB
            yield {
                "chunk_id": chunk_id, 
                "text": chunk, 
                "vector_preview": emb[:3], # Only previewing first 3 dimensions for the console 
                "vector_length": len(emb)
            }

def main():
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("InsurancePolicyProcessor") \
        .master("local[*]") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .getOrCreate()

    hdfs_path = "hdfs://localhost:9000/data/raw/insurance_pdfs/*.pdf"
    print(f"Reading PDFs from: {hdfs_path}")
    
    # 1. Read Binary from HDFS
    pdf_rdd = spark.sparkContext.binaryFiles(hdfs_path)

    # 2. Extract Text
    text_rdd = pdf_rdd.mapValues(extract_text_from_pdf)
    
    # 3. Chunk the Text
    chunked_rdd = text_rdd.mapValues(chunk_text)
    
    # 4. Generate Embeddings (Distributed across worker partitions)
    print("Starting Distributed Chunking and Vectorization...")
    embedded_rdd = chunked_rdd.mapPartitions(generate_embeddings)

    # 5. Fetch and Verify
    print("Processing complete. Fetching top 3 vectorized chunks...")
    results = embedded_rdd.take(3) 
    
    print("\n" + "="*70)
    for res in results:
        print(f"CHUNK ID: {res['chunk_id']}")
        print(f"TEXT PREVIEW: {res['text'][:100].replace(chr(10), ' ')}...")
        print(f"VECTOR DIMENSIONS: {res['vector_length']} (e.g., {res['vector_preview']}...)")
        print("-" * 70)

    spark.stop()

if __name__ == "__main__":
    main()