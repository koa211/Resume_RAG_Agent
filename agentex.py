from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """You are a resume research assistant.

## Capabilities

- `fetch_text_from_txt`: loads document text into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""

@tool 
def fetch_text_from_txt(file: str) -> str:
    """Fetch the txt from file in dir"""
    with open(file, "r", encoding="utf-8") as f:
        return f.read()

model = init_chat_model(
    "openai:gpt-5.4-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_txt],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f"""Read my resume from CS_resume.txt using fetch_text_from_txt."
Based on my listed skills and experience, what should I prioritise learning next given the 
current job market trends? Ground your answer in what's actually in the resume—don't 
guess at skills I don't have."""

agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "resume-talk"}},
)

print(agent_result["messages"][-1].content_blocks)