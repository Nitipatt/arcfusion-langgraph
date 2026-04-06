# ArcFusion LangGraph AI - Agent Notes

This document provides specialized guidelines and context for continuous development within the ArcFusion LangGraph AI architecture.

## Project Structure
- `app/main.py` is the application entry point. 
- Execution flow lives in `app/graph.py` which ties together python functions acting as nodes in `app/nodes/`. 

## Core Principles
- Treat the prompt structure as code. Enhancements to the `generate_insights` or query execution blocks must factor in LLM unpredictability.
- Caching is semantic. Ensure you understand `app/cache.py` configurations regarding similarity thresholds to prevent false positives when testing new query capabilities.
- The service dynamically accesses databases provided in the FastAPI payload. Never hardcode structural SQL logic, as schemas vary inherently by client.

## Coding Guidelines
- All major exceptions within LangGraph nodes should safely format into the defined `AgentState` under `sql_errors` to allow for graceful UX degradation or self-heal retry loops.
- Use Python typing (`Pydantic`) rigidly, particularly when parsing the complex JSON outputs generated from the LLM. Data framing should utilize standard libraries like `pandas` where applicable.
