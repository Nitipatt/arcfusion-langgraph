# ArcFusion LangGraph AI Service

This repository manages the intelligent analytics engine for ArcFusion. It utilizes state-based graph AI reasoning frameworks to receive natural language queries, interact with remote databases safely, execute SQL, and formulate rich graphical insights.

## Comprehensive Architecture Overview

The LangGraph Service is fundamentally designed to be a computationally intensive, stateless (cache-supported) orchestrator isolated from general user traffic.

### Key Technologies
- **Framework**: FastAPI (Entrypoint for inter-service communication)
- **AI Orchestration**: LangGraph, LangChain
- **LLM Interfaces**: Langchain OpenAI / Anthropic integrations
- **Data Engineering**: Pandas (for statistical manipulation and transformation)
- **Security & Caching**: Custom Semantic Cache layer checking threshold limits of previous calculations to save token/compute costs.

### Analytical Pipeline Architecture
1. **API Entry (`app/main.py`)**: Receives the query and database metadata. Instantiates the initial agent state and checks the semantic cache.
2. **Graph Structure (`app/graph.py`)**: Builds the execution acyclic workflow using LangGraph. Handles looping mechanisms for SQL correction on faulty schemas.
3. **Nodes (`app/nodes/`)**:
   - `schema_extractor.py`: Fetches the layout of tables/columns securely.
   - `generate_insights.py`: Formats findings into narrative and title.
   - Other nodes execute SQL natively and validate context limits.
4. **State Management (`app/state.py`)**: Defines the typed contract mapping the agent's memory as it transverses graph nodes. Re-attempts and history persistence are defined here before yielding final output back to the primary API.

## How to Run It

### Prerequisites
- Python 3.10+
- Access to an LLM provider (OpenAI API key)

### Local Development Setup
1. **Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Rename or copy `.env.example` to `.env` and configure your API keys for the LLMs.
   ```env
   OPENAI_API_KEY=sk-...
   SERVICE_PORT=8001
   ```

4. **Start the Engine**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```
   Service health check will be routed to `http://localhost:8001/health`.
