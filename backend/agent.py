# backend/agent.py
import os
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# --- Setup LLM, Embeddings, and Retriever ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
qdrant_client = Qdrant.from_existing_collection(
    path="qdrant_db/",
    collection_name="manuals",
    embedding=embeddings,
)
retriever = qdrant_client.as_retriever()


# --- Define Agent State ---
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str


# --- Define Nodes ---
def retrieve(state):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant for cloud security. 
        Answer the user's question based only on the following context:
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        Provide a clear, step-by-step answer if possible.
        """
    )

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": documents, "question": question})
    return {"generation": generation}


def grade_documents(state):
    print("---CHECKING DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    # A simple relevance check - you could make this more sophisticated
    if not documents:
        print("---DECISION: NO DOCUMENTS FOUND---")
        return "no"

    print("---DECISION: DOCUMENTS FOUND, PROCEEDING TO GENERATE---")
    return "yes"


# --- Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "yes": "generate",
        "no": END,  # Or route to a node that says "I can't find an answer"
    },
)
workflow.add_edge("generate", END)

# Compile the graph
agent = workflow.compile()


# Function to run the agent
def run_agent(question: str):
    result = agent.invoke({"question": question})
    if result.get("generation"):
        return result["generation"]
    else:
        return (
            "I'm sorry, I couldn't find relevant information in the provided manuals."
        )


# Add this to the end of backend/agent.py
if __name__ == "__main__":
    print("--- Running Agent Standalone Test ---")
    # Ask a question you KNOW is in your test PDF
    test_question = (
        "What is risk management?"  # Change this based on your PDF's content
    )
    answer = run_agent(test_question)
    print("\n--- Agent's Answer ---")
    print(answer)
