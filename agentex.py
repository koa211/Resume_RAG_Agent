import chromadb
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv("key.env")

SYSTEM_PROMPT = """You are a resume research assistant.

## Capabilities

- `search_documents: Fetch the txt from a folder that contains resume and job posting`
`search_web: Search the web for current job market trends, salary data, or in-demand skills `"""

folder = "job_collection"
filenames = ["CS_resume.txt", "CS_resume2.txt", "JobPostings.txt", "LinkedinJobs.txt"]
documents = []
for fname in filenames:
    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
        documents.append(f.read())

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="job_collection")

if collection.count() == 0:
    documents = []
    for fname in filenames:
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())
    collection.upsert(ids=["id1", "id2", "id3", "id4"], documents=documents)

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

content = f"""Using the documents that I have."
Based on my listed skills and experience, what should I prioritise learning next given the 
current job market trends? Ground your answer in what's actually in the resume—don't 
guess at skills I don't have."""

agent1 = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    thread_config,
)["messages"][-1].content_blocks
print(agent1)

response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Of those, which one should I start with this week?"}]},
    thread_config,
)["messages"][-1].content_blocks
print(response2)

# asking what I build recently
content2 = "What AI/ML projects have I built recently, based on my resume?"

response3 = agent.invoke(
    {"messages": [{"role": "user", "content": content2}]},
    thread_config,
)["messages"][-1].content_blocks
print(response3)