import os
from typing import Dict, TypedDict, List
from dotenv import load_dotenv
import cohere
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END

# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load Environment Variables
load_dotenv()
cohere_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_key)

# 1. DEFINE THE AGENT STATE
# This holds the memory for our swarm as they pass data to each other
class GraphState(TypedDict):
    question: str
    vector_context: str
    graph_context: str
    final_answer: str

# 2. THE VECTOR RETRIEVER AGENT
def retrieve_from_milvus(state: GraphState) -> Dict:
    print("🤖 Agent 1: Searching Milvus for semantic text chunks...")
    question = state["question"]
    
    # Embed the question
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode([question]).tolist()[0]

    # Search Milvus
    db_client = MilvusClient(uri="tcp://127.0.0.1:19530")
    search_res = db_client.search(
        collection_name="insurance_policies",
        data=[question_embedding],
        limit=3,
        output_fields=["text"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )

    context = ""
    for hit in search_res[0]:
        context += f"- {hit['entity']['text']}\n"
        
    return {"vector_context": context}

# 3. THE GRAPH RETRIEVER AGENT
def retrieve_from_neo4j(state: GraphState) -> Dict:
    print("🤖 Agent 2: Searching Neo4j for logical policy rules...")
    question = state["question"]
    
    # Use Cohere to extract entities from the user's question to search the graph
    extract_prompt = f"Extract the main healthcare scheme, disease, or entity from this question as a single uppercase string. Question: {question}"
    entity_res = co.chat(message=extract_prompt, model='command-r-08-2024')
    search_entity = entity_res.text.strip().upper()
    
    # Query Neo4j for immediate neighbors
    URI = "bolt://127.0.0.1:7687"
    AUTH = ("neo4j", "insurance_graph_password")
    graph_context = f"Known rules for {search_entity}:\n"
    
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                # Find relationships where our entity is the subject OR the object
                query = """
                MATCH (n:Entity)-[r]-(m:Entity) 
                WHERE n.name CONTAINS $entity 
                RETURN n.name, type(r), m.name LIMIT 10
                """
                results = session.run(query, entity=search_entity)
                for record in results:
                    graph_context += f"- {record['n.name']} {record['type(r)']} {record['m.name']}\n"
    except Exception as e:
        graph_context = "Could not retrieve graph rules."

    return {"graph_context": graph_context}

# 4. THE SYNTHESIZER AGENT
def synthesize_answer(state: GraphState) -> Dict:
    print("🤖 Agent 3: Synthesizing final answer from Vector and Graph data...\n")
    
    prompt = f"""You are an expert AI Insurance Architect. 
    Answer the user's question using ONLY the provided Vector Text and Graph Rules.
    
    CRITICAL OUTPUT INSTRUCTIONS:
    If the user's question involves comparing schemes, listing eligibility, or asking "what schemes" are available, you MUST format your final answer as a structured Markdown Table called "The Eligibility Matrix".
    Use these columns (if the data is available): 
    | Scheme Name | Target Beneficiary | Key Benefits / Coverage | Prerequisites |
    
    If the question is a simple definition, you may answer with standard text.
    
    USER QUESTION: {state['question']}
    
    VECTOR TEXT (Milvus):
    {state['vector_context']}
    
    GRAPH RULES (Neo4j):
    {state['graph_context']}
    """
    
    response = co.chat(
        message=prompt,
        model='command-r-08-2024',
        temperature=0.2
    )
    
    return {"final_answer": response.text}

# 5. ORCHESTRATE THE SWARM (LANGGRAPH)
def build_and_run_graph(user_question: str):
    # Initialize the graph
    workflow = StateGraph(GraphState)

    # Add the nodes (our agents)
    workflow.add_node("vector_agent", retrieve_from_milvus)
    workflow.add_node("graph_agent", retrieve_from_neo4j)
    workflow.add_node("synthesizer_agent", synthesize_answer)

    # Define the flow: Start -> Parallel Retrieval -> Synthesis -> End
    workflow.set_entry_point("vector_agent")
    workflow.add_edge("vector_agent", "graph_agent") # In a production system, these run in parallel
    workflow.add_edge("graph_agent", "synthesizer_agent")
    workflow.add_edge("synthesizer_agent", END)

    # Compile the swarm
    app = workflow.compile()
    
    # Run the swarm
    inputs = {"question": user_question}
    print("="*60)
    print(f"👤 USER: {user_question}")
    print("="*60)
    
    for output in app.stream(inputs):
        pass # The nodes handle their own logging

    print("="*60)
    print("✅ FINAL RAG OUTPUT:")
    print(output['synthesizer_agent']['final_answer'])
    print("="*60 + "\n")

if __name__ == "__main__":
    # Test your new swarm! Ask a question related to the PDF you ingested.
    test_q = "What schemes are intended for farmers or agriculture?"
    build_and_run_graph(test_q)