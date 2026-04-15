# Blog Writing Agent

A Streamlit-based blog writing assistant powered by LangGraph and Groq. This project generates structured blog content with planning, optional research, and image planning support using AI models.

## Project Overview

- `bwa_frontend.py` — Streamlit frontend for generating blog posts and downloading results.
- `bwa_backend.py` — LangGraph application logic, including router, research, orchestration, and reducer flows.
- `requirements.txt` — Python dependency list for the project.
- `.env` — local environment variables (not committed to source control).

## Key Features

- Uses Streamlit to provide a web UI for blog generation.
- Supports topic planning, task-based blog sections, and image placeholder planning.
- Uses Groq via `langchain-groq` as the primary LLM backend.
- Includes Markdown normalization and ZIP export helpers.

## Requirements

The project is built for Python 3.12 and depends on packages such as:

- `streamlit`
- `pandas`
- `Pillow`
- `python-dotenv`
- `pydantic`
- `langgraph`
- `langchain`
- `langchain-groq`
- `langchain-core`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Local Setup

1. Activate your virtual environment:

```powershell
.\myenv\Scripts\Activate.ps1
```

2. Install requirements if not already installed:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```text
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

4. Run the Streamlit app:

```powershell
streamlit run bwa_frontend.py
```

5. Open the app in your browser at `http://localhost:8501`.

## Environment Variables

The app expects these values in `.env` or your environment:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN`
- `GROQ_API_KEY`
- `TAVILY_API_KEY`

## Recommended `.gitignore`

The repository should ignore local and sensitive files:

```gitignore
.env
myenv/
PracticeCode/
__pycache__/
*.py[cod]
.streamlit/
Desktop.code-workspace
.vscode/
.idea/
.DS_Store
Thumbs.db
```

## Deployment Notes

- This app is suitable for deployment to services like Streamlit Cloud, Heroku, or any container host.
- Keep API keys out of source control and configure them in your deployment environment.
- If packaging with Docker, use `requirements.txt` and expose Streamlit on port `8501`.

## File Structure

- `bwa_frontend.py` — frontend application entrypoint
- `bwa_backend.py` — core backend logic and LangGraph application
- `requirements.txt` — dependency manifest
- `.env` — environment variables file
- `myenv/` — local Python virtual environment (should be ignored)

## Notes

- Do not commit `.env` or local environment files.
- If you modify the app, restart Streamlit to pick up changes.
- `langgraph` is used for state flow orchestration and structured prompt handling.
