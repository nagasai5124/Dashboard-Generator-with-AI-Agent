# Dashboard-Generator-with-AI-Agent
This project is a Streamlit-based web application designed to transform static data sources like PDFs and CSV files into interactive analytical dashboards. It leverages LLM-based agents (using tools like LangChain and OpenAI/Ollama models) to understand content, generate insights, and visualize data — with no coding required by the user.

# Core Objectives
Enable users to upload PDF and CSV files

Use an AI agent to process the content

Automatically summarize, analyze, and visualize the data

Provide a clean, interactive UI using Streamlit

Allow flexible agent-based interactions using LangChain Agents + Tools like PythonREPLTool

# Architecture
1. File Upload:
Users upload a PDF or CSV file via Streamlit UI.

Content Extraction:

PDFs: Text is extracted using libraries like pypdfloader.

CSVs: Data is parsed using pandas.

2. Agent Initialization:

A LangChain Agent is initialized with tools like PythonREPLTool and LLM (gemma3 from Ollama).

This agent is stored in st.session_state.agent for stateful interaction.

Agent Task Execution:

Generate summaries

Create visualizations (with code execution via REPL)

# output
