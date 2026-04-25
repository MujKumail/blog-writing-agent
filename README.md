# Blog Writing Agent

An AI-powered blog writing app built with `Streamlit` and `LangGraph` that takes a topic, researches it, generates a structured blog draft, and lets you preview or download the result in Markdown and styled PDF format.

## Features

- Topic-based blog generation
- Research-backed writing using external sources
- Structured blog planning and section-wise content generation
- Markdown preview inside the app
- Download blog as `.md`
- Download styled PDF export
- View evidence/source links used during generation
- Load and reuse previously generated blogs

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain
- Groq API
- Tavily API
- Pillow
- Pydantic

## How It Works

1. Enter a blog topic.
2. The agent decides whether research is needed.
3. Relevant sources are collected and summarized.
4. A blog plan is generated.
5. The blog is written section by section.
6. The final content is previewed and can be downloaded as Markdown or PDF.

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd blog-writing-agent
