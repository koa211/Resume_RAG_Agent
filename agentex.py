import chromadb
import os
import hashlib
import json

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv("key.env")

HASH_FILE = "file_hashes.json"

SYSTEM_PROMPT = """You are a resume research assistant.

## Capabilities

- `search_documents: Fetch the txt from a folder that contains resume and job posting`
`search_web: Search the web for current job market trends, salary data, or in-demand skills `"""

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def files_changed(filepaths):
    old_hashes = {}
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            old_hashes = json.load(f)

    new_hashes = {fp: get_file_hash(fp) for fp in filepaths}
    changed = new_hashes != old_hashes

    with open(HASH_FILE, "w") as f:
        json.dump(new_hashes, f)

    return changed

folder = "job_collection"
filenames = ["CS_resume.txt", "CS_resume2.txt", "JobPostings.txt", "LinkedinJobs.txt"]
documents = []
for fname in filenames:
    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
        documents.append(f.read())

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="job_collection")


filepaths = [os.path.join(folder, f) for f in filenames]
needs_embed = collection.count() == 0 or files_changed(filepaths)
if needs_embed:
    print("Re-embedding...")
    documents = []
    for fname in filenames:
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())
    collection.upsert(ids=["id1", "id2", "id3", "id4"], documents=documents)
else:
    print("Using cached embeddings")

results = collection.query(
    query_texts=["These are documents about a resume that is chunked and job postings from linkedin and indeed"],
    n_results=4
)

print(results)
@tool 
def search_documents(query: str) -> str:
    """Fetch the txt from a folder that contains resume and job posting"""
    results = collection.query(query_texts=[query], n_results=4)
    return "\n\n".join(results["documents"][0])

model = init_chat_model(
    "openai:gpt-5.4-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
@tool
def search_web(query: str) -> str:
    """Search the web for current job market trends, salary data, or in-demand skills."""
    response = tavily_client.search(query)
    return "\n\n".join(r["content"] for r in response["results"])

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[search_documents, search_web],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

thread_config = {"configurable": {"thread_id": "resume-talk"}}

while True:
    user_input = input("Ask something (or quit): ")
    if user_input.lower() == "quit":
        break
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        thread_config,
    )["messages"][-1].content_blocks
    print(response)