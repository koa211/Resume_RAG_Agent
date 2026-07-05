# Resume RAG Agent

Searches live job postings and analyses your resume to figure out what skills are actually in demand — grounded in your real experience, not guesses.

## Features

- Multi-doc retrieval over resume + job postings (chunked, embedded, queried via Chroma)
- Live job market search via Tavily
- Multi-turn conversation memory (LangGraph checkpointer)

## Tech stack

- LangChain / LangGraph
- ChromaDB
- Tavily
- OpenAI (gpt-5.4-mini)

## Setup

```bash
git clone <repo-url>
cd resume-rag-agent
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Set environment variables:

```powershell
$env:OPENAI_API_KEY="your-key-here"
$env:TAVILY_API_KEY="your-key-here"
```

## Usage

Place your resume and job posting text files in `job_collection/`, then run:

```bash
python agentex.py
```

Example query:

> "Based on my listed skills and experience, what should I prioritise learning next given current job market trends?"

## Example output

**Q:** Based on my resume, what should I prioritise learning next?

**A:** Kubernetes, CI/CD with GitHub Actions, and Azure — given existing Spring Boot/Docker/AWS experience and repeated mentions of cloud-native deployment across the job postings retrieved.

**Follow-up (same thread):** "Of those, which one should I start with this week?"

**A:** Kubernetes — best overlap with current Docker experience, most frequently mentioned across postings.

## Folder structure

```
job_collection/
  CS_resume.txt       # resume, part 1
  CS_resume2.txt       # resume, part 2
  JobPostings.txt      # scraped job postings
  LinkedinJobs.txt     # LinkedIn postings
```

## Limitations

- Chroma runs in-memory — collection resets every run, no persistence yet
- No deduplication or chunk overlap tuning applied

## Next steps

- [ ] Persistent vector store (Chroma with disk storage, or swap to a hosted option)
- [ ] Add this project itself as a resume entry, then re-test grounding
- [ ] Maybe a simple CLI or web UI instead of hardcoded queries
