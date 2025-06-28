import os
import time
import re
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemma model through Groq
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="gemma2-9b-it"
)

# QA Prompt Function using Groq-Gemma
def qa_prompt_fn(question: str, context: str) -> str:
    prompt = f"""
You are a legal expert AI assistant trained on BHARATIYA NYAYA SANHITA, 2023 (BNS) and landmark case references.

Given the following crime narration and the context extracted from legal documents, your task is to analyze and return the following details with legal accuracy and depth:

1. The identified crime (in simple terms).
2. The most relevant section(s) of Indian Penal Code (IPC), applicable to the crime.
3. Important legal definitions or terms used in the section (e.g., mens rea, actus reus, dishonestly, voluntarily, etc.).
4. Any general or special exceptions (e.g., age, mental capacity, necessity) relevant to the case.
5. Illustrations or examples from IPC that resemble the incident.
6. Potential punishment according to IPC for the identified section.

Respond strictly in the format below:

Crime: [Short title or description]
IPC Section: [e.g., Section 103 - Theft]
Legal Definitions/Terms: [Any important terms from BNS with meaning]
Exceptions: [If any exception from Chapter IV applies]
Illustration (if applicable): [An example close to the case]
Punishment: [Legal punishment under IPC]

If a crime or IPC section cannot be clearly identified, respond with:
"Unable to identify."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    response = llm.invoke(prompt)
    result = response.content if response and hasattr(response, 'content') else "No response"

    # Define regex pattern here
    pattern = (
        r"Crime: .*?\n"
        r"IPC Section: .*?\n"
        r"Legal Definitions/Terms: .*?\n"
        r"Exceptions: .*?\n"
        r"Illustration \(if applicable\): .*?\n"
        r"Punishment: .*"
    )

    # Guardrail check
    if not re.search(pattern, result, re.DOTALL) and "Unable to identify" not in result:
        return "Output format validation failed. No valid response."

    return result

# Function to perform RAG
def query_documents(query: str):
    # Guardrail: Input validation
    if not query.strip() or len(query) > 500:
        return "Invalid query. Please provide a concise and meaningful crime description."

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    combined_context = "\n".join([doc.page_content for doc in docs])
    return qa_prompt_fn(query, combined_context)

# Example
if __name__ == "__main__":
    query = "Which BNS section applies if a person forcibly enters someone's house at night?"
    answer = query_documents(query)
    print("Answer:\n", answer)
