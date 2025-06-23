import streamlit as st
import pandas as pd
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from pandasai import Agent
import pandasai as pai
from datetime import datetime
from pandasai import SmartDatalake
import json

# Page configuration
st.set_page_config(
    page_title="Excel File Chat Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        align-self: flex-start;
        margin-right: 20%;
    }
    .judge-verdict {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .judge-verdict.approved {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .judge-verdict.rejected {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .explanation-section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.3rem;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .message-time {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
    .file-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
    }
    .sheet-info {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #ff9800;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class ExcelChatBot:
    def __init__(self):
        self.setup_api_key()
        self.initialize_models()
        
    def setup_api_key(self):
        """Setup Google API key from environment or user input"""
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            api_key = st.sidebar.text_input(
                "Enter your Google API Key:",
                type="password",
                help="Get your API key from https://aistudio.google.com/"
            )
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                st.sidebar.success("API Key set successfully!")
            else:
                st.sidebar.warning("Please enter your Google API Key to continue.")
                st.stop()
        
    def initialize_models(self):
        """Initialize LangChain and PandasAI models"""
        try:
            # Initialize general chat model
            self.chat_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Initialize judge model (using same model but with different temperature for consistency)
            self.judge_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=0.1,  # Lower temperature for more consistent judging
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Initialize PandasAI LLM
            self.pandas_llm = self.chat_model
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.stop()
    
    def load_excel_files(self, uploaded_files):
        """Load and process multiple Excel files, returning a flattened dict of DataFrames"""
        try:
            all_dataframes = {}
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                # Read all sheets
                excel_file = pd.ExcelFile(tmp_file_path)
                sheet_names = excel_file.sheet_names
                for sheet_name in sheet_names:
                    try:
                        df = pd.read_excel(tmp_file_path, sheet_name=sheet_name)
                        if not df.empty:
                            key = f"{uploaded_file.name} - {sheet_name}"
                            all_dataframes[key] = df
                    except Exception as e:
                        st.warning(f"Could not read sheet '{sheet_name}' from {uploaded_file.name}: {str(e)}")
                        continue
                # Clean up temp file
                os.unlink(tmp_file_path)
            return all_dataframes if all_dataframes else None
        except Exception as e:
            st.error(f"Error loading Excel files: {str(e)}")
            return None
    
    def is_data_related_query(self, query):
        """Check if query is related to data analysis"""
        data_keywords = [
            'data', 'dataframe', 'df', 'table', 'column', 'row', 'show', 'display',
            'plot', 'chart', 'graph', 'analyze', 'analysis', 'statistics', 'stats',
            'count', 'sum', 'average', 'mean', 'median', 'max', 'min', 'correlation',
            'group', 'filter', 'sort', 'unique', 'null', 'missing', 'describe',
            'head', 'tail', 'info', 'shape', 'value_counts', 'groupby', 'sheet',
            'sheets', 'worksheet', 'workbook'
        ]
        
        query_lower = query.lower()
        is_data_related = any(keyword in query_lower for keyword in data_keywords)
        if is_data_related:
            return True
        else:
            # Check if the query is about data analysis using Gemini llm call
            try:
                prompt = "Is this Text related to any type of data analysis, Excel sheets, or data visualization? Strictly answer in 'yes' or 'no' without any additional text: \n Text: " + query
                response = self.chat_model.invoke(prompt)
                if "yes" in response.content.lower():
                    return True
                else:
                    return False
            except Exception as e:
                st.error(f"Error checking query type: {str(e)}")
                return False

    def judge_response_quality(self, query, response, explanation, dataframes_dict):
        """
        Use LLM as a judge to verify the quality and accuracy of the response
        """
        try:
            # Create context about available data
            data_context = ""
            for sheet_name, df in dataframes_dict.items():
                data_context += f"\nSheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns\n"
                data_context += f"Columns: {', '.join(df.columns.tolist())}\n"
                # Add sample data for first few rows
                data_context += f"Sample data (first 3 rows):\n{df.head(3).to_string()}\n"
                data_context += "-" * 50 + "\n"
            
            # Format response content for judge
            if isinstance(response, dict):
                if response.get("type") == "dataframe":
                    response_content = f"DataFrame with {len(response['content'])} rows and {len(response['content'].columns)} columns:\n{response['content'].to_string()}"
                else:
                    response_content = str(response.get("content", ""))
            else:
                response_content = str(response)
            
            judge_prompt = f"""
    You are an expert data analyst judge. Your task is to evaluate the quality and accuracy of a data analysis response.

    **CONTEXT - Available Data:**
    {data_context}

    **USER QUERY:**
    {query}

    **AI RESPONSE:**
    {response_content}

    **AI EXPLANATION OF APPROACH:**
    {explanation}

    **EVALUATION CRITERIA:**
    Please evaluate the response based on:
    1. **Accuracy**: Does the response correctly address the user's query?
    2. **Completeness**: Does the response provide sufficient information?
    3. **Data Usage**: Does the response appropriately use the available data?
    4. **Methodology**: Is the analytical approach sound and well-explained?
    5. **Clarity**: Is the response clear and understandable?

    **REQUIRED OUTPUT FORMAT:**
    Respond with ONLY a valid JSON object (no markdown formatting, no code blocks, no additional text):

    {{
        "verdict": "APPROVED" or "REJECTED",
        "confidence_score": <number between 0.0 and 1.0>,
        "reasoning": "<detailed explanation of your judgment>",
        "issues_found": ["<list of specific issues if any>"],
        "suggestions": ["<list of improvement suggestions if any>"]
    }}

    **IMPORTANT NOTES:**
    - APPROVED: Response is accurate, complete, and properly addresses the query
    - REJECTED: Response has significant issues, inaccuracies, or doesn't address the query properly
    - Be strict but fair in your evaluation
    - Consider both the response content and the explanation of methodology
    - Provide specific, actionable feedback in reasoning and suggestions
    - Return ONLY the JSON object without any markdown formatting or code blocks
    """

            # Get judgment from LLM
            judge_response = self.judge_model.invoke(judge_prompt)
            
            # Clean the response content to handle markdown code blocks
            response_content = judge_response.content.strip()
            
            # More robust cleaning of markdown code blocks using regex
            import re
            
            # Remove markdown code blocks - handles ```json, ```, and variations
            response_content = re.sub(r'^```(?:json)?\s*\n?', '', response_content, flags=re.MULTILINE)
            response_content = re.sub(r'\n?```[ \t]*$', '', response_content, flags=re.MULTILINE)
            
            # Additional cleanup for any remaining backticks or whitespace
            response_content = response_content.strip('`').strip()
            
            # Parse JSON response
            try:
                judgment = json.loads(response_content)
                
                # Validate required fields and set defaults if missing
                if "verdict" not in judgment:
                    judgment["verdict"] = "UNKNOWN"
                if "confidence_score" not in judgment:
                    judgment["confidence_score"] = 0.5
                if "reasoning" not in judgment:
                    judgment["reasoning"] = "No reasoning provided"
                if "issues_found" not in judgment:
                    judgment["issues_found"] = []
                if "suggestions" not in judgment:
                    judgment["suggestions"] = []
                
                # Ensure confidence_score is between 0.0 and 1.0
                confidence = judgment["confidence_score"]
                if confidence > 1.0:
                    judgment["confidence_score"] = 1.0
                elif confidence < 0.0:
                    judgment["confidence_score"] = 0.0
                    
                return judgment
                
            except json.JSONDecodeError as e:
                # Enhanced fallback with more detailed error info
                return {
                    "verdict": "PARSING_ERROR",
                    "confidence_score": 0.0,
                    "reasoning": f"Failed to parse judge response as JSON. Error: {str(e)}. Raw response: {response_content[:200]}...",
                    "issues_found": [f"JSON parsing error: {str(e)}"],
                    "suggestions": ["Manual review required due to JSON parsing failure"]
                }
                
        except Exception as e:
            return {
                "verdict": "SYSTEM_ERROR",
                "confidence_score": 0.0,
                "reasoning": f"Error during judgment process: {str(e)}",
                "issues_found": [f"Judge system error: {str(e)}"],
                "suggestions": ["Manual review required due to system error"]
            }    
    def chat_with_data(self, agent: Agent, query, dataframes_dict: dict):
            """Chat with Excel data using PandasAI with multiple dataframes and LLM judge verification"""
            
            # Check if agent is None
            if agent is None:
                return {
                    "type": "text", 
                    "content": "AI agent is not available. Please reload your file to initialize the agent."
                }
            
            try:
                # Get response from PandasAI agent
                response = agent.chat(query)
                
                # Handle case when PandasAI returns None or empty response
                if response is None or (isinstance(response, str) and response.strip() == ""):
                    fallback_response = self.generate_fallback_response(dataframes_dict, query)
                    return {"type": "text", "content": fallback_response}
                
                # Get explanation from agent
                explanation = agent.explain()
                print("*" * 100)
                print("Explanation:", explanation)
                print("*" * 100)
                
                # Prepare response object based on type
                if isinstance(response, pd.DataFrame):
                    response_obj = {"type": "dataframe", "content": response}
                elif hasattr(response, 'savefig'):  # matplotlib figure
                    response_obj = {"type": "plot", "content": response}
                elif isinstance(response, (list, dict, int, float)):
                    response_obj = {"type": "text", "content": str(response)}
                else:
                    response_obj = {"type": "text", "content": str(response)}
                
                # Get judgment from LLM judge
                judgment = self.judge_response_quality(query, response_obj, explanation, dataframes_dict)
                
                # Add judgment and explanation to response
                response_obj["explanation"] = explanation
                response_obj["judgment"] = judgment
                
                # Log judgment results
                print("*" * 100)
                print(f"JUDGE VERDICT: {judgment.get('verdict', 'UNKNOWN')}")
                print(f"CONFIDENCE: {judgment.get('confidence_score', 0.0)}")
                print(f"REASONING: {judgment.get('reasoning', 'No reasoning provided')}")
                print("*" * 100)
                
                return response_obj
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error in chat_with_data: {error_msg}")
                
                # Generate fallback response when PandasAI fails
                fallback_response = self.generate_fallback_response(dataframes_dict, query)
                return {
                    "type": "text", 
                    "content": f"I encountered an issue processing your data query. Here's what I can tell you based on your data structure:\n\n{fallback_response}",
                    "explanation": "Fallback response due to PandasAI error",
                    "judgment": {
                        "verdict": "ERROR",
                        "confidence_score": 0.0,
                        "reasoning": f"PandasAI error: {error_msg}",
                        "issues_found": [error_msg],
                        "suggestions": ["Try rephrasing your query or check data format"]
                    }
                }

    def generate_fallback_response(self, dataframes_dict, query):
        """Generate fallback response using LLM when PandasAI fails"""
        try:
            # Create summary of all sheets
            sheets_summary = []
            for sheet_name, df in dataframes_dict.items():
                summary = f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns"
                summary += f", columns: {', '.join(df.columns.tolist())}"
                sheets_summary.append(summary)
            
            context = "Excel file contains the following sheets:\n" + "\n".join(sheets_summary)
            
            prompt = f"""
            Based on this Excel file context:
            {context}
            
            User Query: {query}
            
            Please provide a helpful response about what analysis could be done or what information might be available in this data. 
            If the query asks for specific data operations, suggest how the user might approach it.
            Be specific and actionable in your response.
            """
            
            response = self.chat_model.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"I apologize, but I couldn't process your data-related query at this time. Error: {str(e)}"
    
    def general_chat(self, query):
        """General chat using LangChain"""
        try:
            response = self.chat_model.invoke(query)
            return {"type": "text", "content": response.content}
            
        except Exception as e:
            return {"type": "text", "content": f"Error in general chat: {str(e)}"}

    def get_data_summary(self, dataframes_dict):
        """Get summary of all sheets in uploaded dataframes"""
        total_rows = 0
        total_columns = 0
        total_memory = 0
        sheets_info = {}
        for key, df in dataframes_dict.items():
            rows = len(df)
            cols = len(df.columns)
            sheets_info[key] = {
                "rows": rows,
                "columns": cols,
                "column_names": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            total_rows += rows
            total_columns += cols
            total_memory += sheets_info[key]["memory_usage"]
        return {
            "total_sheets": len(dataframes_dict),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_memory_usage": total_memory,
            "sheets_info": sheets_info
        }
        
    def load_agent(self, dataframes_dict) -> Agent:
        """Load PandasAI agent with multiple dataframes"""
        dataframes_list = []
        descriptions = []
        for key, df in dataframes_dict.items():
            dataframes_list.append(df)
            # key = "filename - sheetname"
            filename, sheet_name = key.split(' - ', 1)
            # Full columns list, no truncation
            cols = ', '.join(df.columns.tolist())
            desc = f"Sheet/Dataframe '{sheet_name}': Contains {len(df)} rows and {len(df.columns)} columns. Columns: {cols}."
            descriptions.append(desc)
        
        excel_files_info = ""
        
        print("*"*150)
        print(excel_files_info)
        print("*"*150)
   
        # excel_files_info += new_file_descriptions

        combined_description = f"""# Business Analysis Agent Instructions

You are an AI assistant specialized in analysis and data interpretation.

You have access to business knowledge and the ability to analyze data from the following data composition:

<data_info>
{excel_files_info}
</data_info>

Your role: Data Analyst

**CRITICAL FOR EFFICIENCY**: For simple, direct queries (like "what is the budget for a specific month"), provide a CONCISE response without extensive analysis blocks. Use the streamlined approach below:

**For Simple Budget/Data Queries:**
1. Directly identify the relevant file and data needed from the schema
2. Use the column structures provided in the <data_info> schema
3. Write efficient Python code using the schema information - DO NOT write exploratory code
4. Execute once and provide the answer
5. Skip extensive planning blocks for straightforward requests

**For Complex Analysis Only:**
Use the full analysis and planning structure with 5-point breakdown.

**IMPORTANT**: Use the schema documentation provided in <data_info> to identify exact column names and sheet structures. Do not write code to explore or discover the file structure.

Always be direct and efficient. Avoid repetitive debugging unless absolutely necessary. """

        agent = Agent(
            dataframes_list, 
            config={"llm": self.pandas_llm, "verbose": True, "save_logs": True},
            description=combined_description,
            memory_size=6 # conversation history pairs to be included set to 3
        )    
        return agent

def display_message(message):
    """Display a message in the chat interface with judge verdict and explanation"""
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle different response types
        if isinstance(message["content"], dict):
            # Display the main content
            if message["content"]["type"] == "dataframe":
                # Display DataFrame using st.dataframe
                st.dataframe(message["content"]["content"], use_container_width=True)
            else:
                # Display text content
                st.markdown(message["content"]["content"])
            
            # Display explanation if available
            if "explanation" in message["content"] and message["content"]["explanation"]:
                with st.expander("🔍 Analysis Explanation", expanded=False):
                    st.markdown(f"""
                    <div class="explanation-section">
                        <strong>How this analysis was performed:</strong><br>
                        {message["content"]["explanation"]}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display judge verdict if available
            if "judgment" in message["content"] and message["content"]["judgment"]:
                judgment = message["content"]["judgment"]
                verdict = judgment.get("verdict", "UNKNOWN")
                confidence = judgment.get("confidence_score", 0.0)
                reasoning = judgment.get("reasoning", "No reasoning provided")
                
                # Set verdict styling
                verdict_class = ""
                verdict_icon = ""
                if verdict == "APPROVED":
                    verdict_class = "approved"
                    verdict_icon = "✅"
                elif verdict == "REJECTED":
                    verdict_class = "rejected"
                    verdict_icon = "❌"
                elif verdict == "ERROR":
                    verdict_class = "rejected"
                    verdict_icon = "⚠️"
                else:
                    verdict_icon = "❓"
                
                st.markdown(f"""
                <div class="judge-verdict {verdict_class}">
                    <strong>{verdict_icon} AI Judge Verdict: {verdict}</strong> 
                    (Confidence: {confidence:.1%})<br>
                    <em>{reasoning}</em>
                </div>
                """, unsafe_allow_html=True)
                
                # Show issues and suggestions if available
                if judgment.get("issues_found") or judgment.get("suggestions"):
                    with st.expander("🔎 Detailed Judge Feedback", expanded=False):
                        if judgment.get("issues_found"):
                            st.markdown("**Issues Found:**")
                            for issue in judgment["issues_found"]:
                                st.markdown(f"• {issue}")
                        
                        if judgment.get("suggestions"):
                            st.markdown("**Suggestions for Improvement:**")
                            for suggestion in judgment["suggestions"]:
                                st.markdown(f"• {suggestion}")
        else:
            # Handle legacy string responses
            st.markdown(str(message["content"]))

def main():
    st.title("📊 Multi-Sheet Excel Chat Assistant with AI Judge")
    st.markdown("Chat with your Excel files and get AI-verified insights from multiple sheets!")
    
    # Initialize chatbot
    chatbot = ExcelChatBot()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = None
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_loading" not in st.session_state:
        st.session_state.agent_loading = False
        
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_files = st.file_uploader(
            "Upload your Excel file(s)",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload one or more Excel files to start chatting with your data"
        )
        if uploaded_files:
            names = [f.name for f in uploaded_files]
            if (st.session_state.file_info is None or
                set(names) != set(st.session_state.file_info.get("filenames", []))):
                with st.spinner("Loading Excel files and all sheets..."):
                    dataframes = chatbot.load_excel_files(uploaded_files)
                    if dataframes:
                        st.session_state.dataframes = dataframes
                        st.session_state.file_info = {
                            "filenames": names,
                            "summary": chatbot.get_data_summary(dataframes)
                        }
                        st.success(f"Loaded {len(names)} file(s) with {len(dataframes)} total sheets.")
                        st.session_state.agent = None
                        st.session_state.agent_loading = False
                        
                # Load agent only if dataframes exist and agent is not already loaded
                if st.session_state.dataframes is not None and st.session_state.agent is None and not st.session_state.agent_loading:
                    st.session_state.agent_loading = True
                    with st.spinner("Initializing AI agent for data analysis..."):
                        try:
                            st.session_state.agent = chatbot.load_agent(st.session_state.dataframes)
                            st.success("AI agent loaded successfully!")
                            st.session_state.agent_loading = False
                        except Exception as e:
                            st.error(f"Error loading agent: {str(e)}")
                            st.session_state.agent = None
                            st.session_state.agent_loading = False

        # Display file information
        if st.session_state.file_info:
            info = st.session_state.file_info["summary"]
            st.info(f"**Files:** {', '.join(st.session_state.file_info['filenames'])}\n" +
                    f"**Total Sheets:** {info['total_sheets']}  |  " +
                    f"**Total Rows:** {info['total_rows']:,}  |  " +
                    f"**Total Columns:** {info['total_columns']}  |  " +
                    f"**Memory:** {info['total_memory_usage']/1024:.1f} KB")
            
            if st.session_state.agent:
                st.success("🤖 AI Agent: Ready")
                st.success("⚖️ AI Judge: Ready")
            elif st.session_state.agent_loading:
                st.warning("🔄 AI Agent: Loading...")
            else:
                st.warning("⚠️ AI Agent: Not loaded")

            st.subheader("📄 Sheet Details")
            for key, sheet_info in info['sheets_info'].items():
                with st.expander(f"{key}"):
                    st.write(f"**Rows:** {sheet_info['rows']:,}")
                    st.write(f"**Columns:** {sheet_info['columns']}")
                    st.write(f"**Memory:** {sheet_info['memory_usage']/1024:.1f} KB")
                    st.write("**Column Details:**")
                    for col, dtype in sheet_info['data_types'].items():
                        missing = sheet_info['missing_values'][col]
                        st.write(f"• **{col}:** {dtype} ({missing} missing)")

        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(message)
        
        # Chat input - only enable if no data is loaded OR if agent is ready
        chat_disabled = False
        placeholder_text = "Ask me anything about your data or general questions..."
        
        if st.session_state.dataframes is not None:
            if st.session_state.agent is None and not st.session_state.agent_loading:
                chat_disabled = True
                placeholder_text = "Please wait for the AI agent to load before asking data questions..."
            elif st.session_state.agent_loading:
                chat_disabled = True
                placeholder_text = "AI agent is loading, please wait..."
        
        user_input = st.chat_input(
            placeholder_text,
            disabled=chat_disabled
        )
        
        if user_input:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
            })
            
            # Process the query
            with st.spinner("Processing with AI verification..."):
                # Check if this is a data-related query and if we have data loaded
                if (st.session_state.dataframes is not None and 
                    st.session_state.agent is not None and 
                    chatbot.is_data_related_query(user_input)):
                    # Data-related query with agent available
                    response = chatbot.chat_with_data(st.session_state.agent, user_input, st.session_state.dataframes)
                else:
                    # General query or no data/agent available
                    if st.session_state.dataframes is not None and chatbot.is_data_related_query(user_input) and st.session_state.agent is None:
                        # Data query but agent not available
                        response = {
                            "type": "text", 
                            "content": "I'm sorry, but the AI agent for data analysis is not available. Please try reloading your file or ask a general question instead."
                        }
                    else:
                        response = chatbot.general_chat(user_input)
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
            
            st.rerun()
    
    with col2:
        # Sample queries
        st.subheader("💡 Sample Queries")
        
        # Different sample queries based on whether data is loaded
        if st.session_state.dataframes is not None and st.session_state.agent is not None:
            sample_queries = [
                "What is my budget by segment ?",
                "What events are happening in June?",
                "What is the average Occupancy On Books This Year from 1st Jan 2025 to 1st May",
                "what are total and unique market segments? Give me a list of all market segments",
                "What is Revenue OTB for August versus STLY?",
                "What is the current Occupancy for September OTB versus STLY?",
                "What is the current ADR OTB for each month in 2025 compared to STLY last year?",
                "What is the current Occupancy % OTB for each month in 2025 compared to STLY last year?",
                "What are the rooms sold OTB for this year 2025 vs STLY?",
                "What is RevPar for May OTB vs STLY?"     
            ]
        else:
            sample_queries = [
                "What's the weather like today?",
                "Explain machine learning",
                "What is quantum physics?",
            ]
                
        for query in sample_queries:
            if st.button(f"'{query}'", key=f"sample_{query}"):
                # Check if we can process this query
                can_process = True
                if st.session_state.dataframes is not None and chatbot.is_data_related_query(query):
                    if st.session_state.agent is None:
                        can_process = False
                
                if can_process:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query,
                    })
                    
                    # Process the sample query
                    with st.spinner("Processing with AI verification..."):
                        if (st.session_state.dataframes is not None and 
                            st.session_state.agent is not None and 
                            chatbot.is_data_related_query(query)):
                            response = chatbot.chat_with_data(st.session_state.agent, query, st.session_state.dataframes)
                        else:
                            response = chatbot.general_chat(query)
                    
                    # Create assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                    }
                    
                    # Add to session state
                    st.session_state.messages.append(assistant_message)
                    
                    # Display the assistant response immediately
                    display_message(assistant_message)
                    
                    st.rerun()
                else:
                    st.warning("Please wait for the AI agent to load before trying data queries.")

if __name__ == "__main__":
    main()