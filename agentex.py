import chromadb
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """You are a resume research assistant.

## Capabilities

- `fetch_text_from_txt`: loads document text into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""

folder = "job_collection"
filenames = ["CS_resume.txt", "CS_resume2.txt", "JobPostings.txt", "LinkedinJobs.txt"]
documents = []
for fname in filenames:
    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
        documents.append(f.read())

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="job_collection")

collection.upsert(
    ids=["id1", "id2", "id3", "id4"],
    documents=documents
)

results = collection.query(
    query_texts=["These are documents about a resume that is chunked and job postings from linkedin and indeed"],
    n_results=4
)

print(results)
@tool 
def search_documents(query: str) -> str:
    """Fetch the txt from file in dir"""
    results = collection.query(query_texts=[query], n_results=4)
    return "\n\n".join(results["documents"][0])

model = init_chat_model(
    "openai:gpt-5.4-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f"""Using the documents that I have."
Based on my listed skills and experience, what should I prioritise learning next given the 
current job market trends? Ground your answer in what's actually in the resume—don't 
guess at skills I don't have."""

agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "resume-talk"}},
)

print(agent_result["messages"][-1].content_blocks)