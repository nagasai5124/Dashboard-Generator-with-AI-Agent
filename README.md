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
![Media Player 5_16_2025 10_53_15 AM](https://github.com/user-attachments/assets/0cee368e-69a3-420d-abe8-98b78d2ff33e)
![Media Player 5_16_2025 10_52_04 AM](https://github.com/user-attachments/assets/15fb100f-88ed-4b08-8ee5-abbc870f771a)
![Media Player 5_16_2025 10_52_07 AM](https://github.com/user-attachments/assets/7345808c-daa2-491d-8bb2-9e0b34998272)
![Media Player 5_16_2025 10_52_11 AM](https://github.com/user-attachments/assets/a37f353c-2747-49d3-b271-a652b28b750c)
![Media Player 5_16_2025 10_52_14 AM](https://github.com/user-attachments/assets/cc005e88-6a04-4d40-8a1e-18ef4f5e13db)
![Media Player 5_16_2025 10_53_36 AM](https://github.com/user-attachments/assets/2fd1e3d4-cff4-4fb3-a13e-7310c95ce528)
![Media Player 5_16_2025 10_53_39 AM](https://github.com/user-attachments/assets/af2afbc0-7cdc-4546-b7a3-5512f46a7a81)


