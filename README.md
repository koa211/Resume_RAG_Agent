# Resume RAG Agent

Searches live job postings and analyses your resume to figure out what skills are actually in demand — grounded in your real experience, not guesses.

## Features

- Multi-doc retrieval over resume + job postings (chunked, embedded, queried via Chroma)
- Live job market search via Tavily
- Multi-turn conversation memory (LangGraph checkpointer)
- Persistent vector storage (Chroma on-disk, survives across runs)

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

Create a `.env` file in the project root (gitignored):

```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
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

**Grounding test:** "What AI/ML projects have I built recently, based on my resume?"

**A:** Correctly identified the Resume RAG Agent itself (LangChain, LangGraph, ChromaDB, Tavily, OpenAI) as the primary AI/ML project, distinguishing it from adjacent automation work (Salesforce Apex triggers) rather than over-classifying everything as ML.

## Folder structure

```
job_collection/
  CS_resume.txt       # resume, part 1
  CS_resume2.txt       # resume, part 2
  JobPostings.txt      # scraped job postings
  LinkedinJobs.txt     # LinkedIn postings
```

## Limitations

- No deduplication or chunk overlap tuning applied
- Re-embeds source docs only when collection is empty — doesn't yet detect changed file content on a populated collection

## Next steps

- [x] Persistent vector store (Chroma with disk storage)
- [x] Add this project itself as a resume entry, then re-test grounding
- [x] Detect changed source files and re-embed only what's updated, instead of empty-collection-only check
- [x] Maybe a simple CLI or web UI instead of hardcoded queries
