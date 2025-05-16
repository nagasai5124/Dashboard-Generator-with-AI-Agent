import streamlit as st
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from typing import List
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.tools import PythonREPLTool
import os
from langchain_experimental.agents import create_pandas_dataframe_agent

model=OllamaLLM(model="gemma3")
# Create a Streamlit app
st.title("dashboard generater")
data=st.file_uploader("upload data",type="pdf")
table=st.file_uploader("upload table",type="csv")
submit=st.button("submit")

placeholder = st.empty()
prompt_template = """
You are an AI assistant. Given the following text data, perform the following tasks:
1. Please generate a comprehensive summary of the provided data.Ensure that the summary encapsulates all key information, including but not limited to: main themes, significant figures, relevant dates, and any critical insights or conclusions drawn from the data.The summary should be concise yet informative, maintaining clarity and coherence while reflecting the entirety of the content.Aim for a length of approximately 200-300 words, and format the summary in a structured manner with bullet points for easy readability, so ensure your summary is relevant and accurate
2. give a Title for the following data
3. List of topics which are covered in the data
4. You are an AI tasked with providing a structured approach to solving a given problem statement.Follow these steps meticulously: 1.**Understand the Problem:** - Clearly define the problem statement.- Identify the key components, variables, and constraints involved.- Ask clarifying questions if necessary to ensure comprehension.2.**Research and Gather Information:** - Identify relevant data, resources, or literature that could inform the solution.- Look for similar problems and their solutions to draw inspiration.3.**Break Down the Problem:** - Divide the problem into smaller, manageable parts or steps.- For each part, outline the specific goals and objectives.4.**Develop Potential Solutions:** - Brainstorm multiple solution strategies for each part of the problem.- Consider the pros and cons of each approach.- Use frameworks or methodologies relevant to the domain (e.g., SWOT analysis, root cause analysis).5.**Evaluate Solutions:** - Assess the feasibility, effectiveness, and potential impact of each proposed solution.- Use criteria such as resources required, time constraints, and stakeholder implications.6.**Create an Action Plan:** - Select the most viable solution(s) based on the evaluation.- Develop a step-by-step action plan for implementation, including timelines and responsibilities.7.**Implement the Solution:** - Execute the action plan while monitoring progress.- Ensure that all stakeholders are informed and engaged throughout the process.8.**Review and Reflect:** - After implementation, evaluate the outcome against the initial problem statement.- Identify lessons learned and areas for improvement.- Document the findings for future reference.9.**Iterate as Necessary:** - If the problem is not resolved or new issues arise, revisit the earlier steps.- Adapt and refine the approach based on feedback and results.The goal is to create a comprehensive, logical, and actionable framework that can be adapted to various problem statements across different contexts.

--- START OF TEXT ---
{text}
--- END OF TEXT ---

Format your response as:

Title:
<Your title>

Summary:
- Point 1
- Point 2
...

Topics:
- Topic 1
- Topic 2
...

Approach:
Step 1: ...
Step 2: ...
"""

if submit:
    # Read the uploaded data
    if data is not None:
        placeholder.success("analyzing data..")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(data.read())
            tmp_path = tmp_file.name

        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        final_prompt = prompt_template.format(text=full_text)
        result = model.invoke(final_prompt)
        # Create a prompt template
        st.subheader("📌 Data Summary and Approach")
        placeholder.empty() 
        st.write(result)

    else:
        st.write("⚠️ No PDF data uploaded.")

    if table is not None:
        placeholder.success("analyzing table..")
        # Read the uploaded table
        df = pd.read_csv(table)
        st.session_state.df = df
        st.session_state.agent = create_pandas_dataframe_agent(
            llm=model,
            df=df,
            verbose=True,
            handle_parsing_errors=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_dangerous_code=True,
        )
        st.session_state.data_uploaded = True
        st.subheader("📊 Table Preview:")
        st.dataframe(st.session_state.df)
        st.subheader("🧮 Null Values in Table:")
        st.write(st.session_state.df.isnull().sum())
        st.subheader("📈 AI Insight:")
        promt="""
            As a data analysis expert, analyze this dataset and provide a comprehensive insight report. Your analysis should be detailed, actionable, and follow this structured format:

            1. Dataset Overview (Basic Properties):
               - Total number of records and features
               - Data types of each column
               - Memory usage and dataset size
               - Completeness score (percentage of non-null values)

            2. Statistical Analysis:
               - Key statistics (mean, median, mode, std dev) for numerical columns
               - Quartile analysis and IQR for detecting outliers
               - Distribution characteristics (skewness, kurtosis)
               - Confidence intervals for key metrics

            3. Feature Deep Dive:
               Numerical Features:
               - Range, variance, and coefficient of variation
               - Outlier detection with z-score and IQR method
               - Distribution shape (normal, skewed, bimodal)
               
               Categorical Features:
               - Unique value counts and proportions
               - Mode and frequency analysis
               - Cardinality assessment
               - Missing value patterns

            4. Relationship Analysis:
               - Correlation matrix with significance levels
               - Chi-square tests for categorical relationships
               - ANOVA for categorical-numerical relationships
               - Feature importance rankings

            5. Business-Focused Insights:
               - Key performance indicators (KPIs)
               - Trend identification and pattern recognition
               - Anomaly detection and potential data quality issues
               - Growth opportunities and risk factors

            6. Actionable Recommendations:
               - Data quality improvement suggestions
               - Feature engineering opportunities
               - Potential modeling approaches
               - Business strategy implications
               - Suggested next steps for deeper analysis

            7. Visualization Recommendations:
               - Specific chart types for each insight
               - Key metrics to highlight
               - Interactive dashboard elements
               - Storytelling flow suggestions

            Based on the dataset {df}, provide a detailed analysis following this structure.
            Format your response with clear headings, bullet points, and emphasis on actionable insights.
            Include confidence levels for statistical findings where applicable.
            """
        prompt_text = promt.format(df=st.session_state.df.to_dict(orient="records"))  # fill in your actual variable(s)
        result=model.invoke(prompt_text)
        st.write(result)
        placeholder.empty()
        placeholder.success("generating dashboard code....")
        
        # Run the agent
        agent_response = st.session_state.agent.run("""
        Analyze the dataset and for each column (except the target column), determine its relationship with the target column.
        
        For each column, provide your response in this exact format:
        Column: [column_name]
        Type: [numerical/categorical]
        Suggested Plot: [plot_type] for [column_name] vs target
        
        Example output:
        Column: age
        Type: numerical
        Suggested Plot: scatter plot for age vs target
        
        Column: gender
        Type: categorical
        Suggested Plot: bar plot for gender vs target
        
        Please analyze each column and provide suggestions in this exact format.
        """)
        st.write(agent_response)

        # Function to create appropriate plots based on data type
        def generate_plots(df):
            st.subheader("📊 Visualizations")
            
            # Get numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Create plots for each feature
            for col in df.columns:
                if col != 'target':  # Skip the target column itself
                    st.write(f"### Analysis of {col}")
                    
                    try:
                        if col in numeric_cols:
                            # For numeric features
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                            
                            # Box plot
                            sn.boxplot(data=df, x='target', y=col, ax=ax1)
                            ax1.set_title(f'Box Plot: {col} by Target')
                            
                            # Distribution plot
                            sn.histplot(data=df, x=col, hue='target', ax=ax2, multiple="layer", alpha=0.5)
                            ax2.set_title(f'Distribution of {col} by Target')
                            
                        else:
                            # For categorical features
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sn.countplot(data=df, x=col, hue='target', ax=ax)
                            ax.set_title(f'Count Plot: {col} by Target')
                            plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Add correlation info for numeric columns
                        if col in numeric_cols:
                            correlation = df[col].corr(df['target'])
                            st.write(f"Correlation with target: {correlation:.2f}")
                            
                    except Exception as e:
                        st.write(f"Could not generate plot for {col}: {str(e)}")
                        continue

        # Generate all plots
        if 'target' in st.session_state.df.columns:
            generate_plots(st.session_state.df)
        else:
            st.warning("Please ensure your dataset has a 'target' column for visualization")
        
        placeholder.empty()
    else:
        st.write("⚠️ No csv data uploaded.")

        
        