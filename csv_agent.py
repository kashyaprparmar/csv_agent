import streamlit as st
import pandas as pd
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from pandasai import Agent
import pandasai as pai
from datetime import datetime

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
            
            # Initialize PandasAI LLM
            self.pandas_llm = self.chat_model
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.stop()
    
    def load_excel_file(self, uploaded_file):
        """Load and process Excel file with multiple sheets"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Read all sheets from Excel file
            excel_file = pd.ExcelFile(tmp_file_path)
            sheet_names = excel_file.sheet_names
            
            # Create dictionary of dataframes for each sheet
            dataframes = {}
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(tmp_file_path, sheet_name=sheet_name)
                    # Only include sheets with data
                    if not df.empty:
                        dataframes[sheet_name] = df
                except Exception as e:
                    st.warning(f"Could not read sheet '{sheet_name}': {str(e)}")
                    continue
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return dataframes if dataframes else None
            
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
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
    
    def chat_with_data(self, dataframes_dict, query):
        """Chat with Excel data using PandasAI with multiple dataframes"""
        try:
            # Prepare dataframes list and descriptions
            dataframes_list = []
            descriptions = []
            
            for sheet_name, df in dataframes_dict.items():
                dataframes_list.append(df)
                # Create description for each sheet
                description = f"Sheet '{sheet_name}': Contains {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}."
                descriptions.append(description)
            
            # Combine descriptions
            combined_description = "Excel file with multiple sheets: " + " | ".join(descriptions)
            
            # Create PandasAI agent with multiple dataframes
            agent = Agent(
                dataframes_list, 
                config={"llm": self.pandas_llm, 'verbose':True},
                description=combined_description
            )
            
            # Get response from PandasAI
            response = agent.chat(query)
            
            # Handle case when PandasAI returns None or empty response
            if response is None or (isinstance(response, str) and response.strip() == ""):
                fallback_response = self.generate_fallback_response(dataframes_dict, query)
                return {"type": "text", "content": fallback_response}
            
            # Check if response is a DataFrame
            if isinstance(response, pd.DataFrame):
                return {"type": "dataframe", "content": response}
            else:
                return {"type": "text", "content": str(response)}
            
        except Exception as e:
            # Generate fallback response when PandasAI fails
            fallback_response = self.generate_fallback_response(dataframes_dict, query)
            return {"type": "text", "content": f"PandasAI encountered an issue: {str(e)}\n\nFallback response:\n{fallback_response}"}
    
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
        """Get summary of all sheets in the uploaded data"""
        total_rows = 0
        total_columns = 0
        total_memory = 0
        sheets_info = {}
        
        for sheet_name, df in dataframes_dict.items():
            sheet_info = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            sheets_info[sheet_name] = sheet_info
            
            total_rows += sheet_info["rows"]
            total_columns += sheet_info["columns"]
            total_memory += sheet_info["memory_usage"]
        
        summary = {
            "total_sheets": len(dataframes_dict),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_memory_usage": total_memory,
            "sheets_info": sheets_info
        }
        return summary

def display_message(message):
    """Display a message in the chat interface"""
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
            if message["content"]["type"] == "dataframe":
                # Display DataFrame using st.dataframe
                st.dataframe(message["content"]["content"], use_container_width=True)
            else:
                # Display text content
                st.markdown(message["content"]["content"])
        else:
            # Handle legacy string responses
            st.markdown(str(message["content"]))

def main():
    st.title("📊 Multi-Sheet Excel Chat Assistant")
    st.markdown("Chat with your Excel files and get instant insights from multiple sheets!")
    
    # Initialize chatbot
    chatbot = ExcelChatBot()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = None
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload your Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file to start chatting with your data"
        )
        
        if uploaded_file is not None:
            if st.session_state.dataframes is None or st.session_state.file_info["filename"] != uploaded_file.name:
                with st.spinner("Loading Excel file and all sheets..."):
                    dataframes = chatbot.load_excel_file(uploaded_file)
                    if dataframes is not None:
                        st.session_state.dataframes = dataframes
                        st.session_state.file_info = {
                            "filename": uploaded_file.name,
                            "summary": chatbot.get_data_summary(dataframes)
                        }
                        st.success(f"File loaded successfully! Found {len(dataframes)} sheets.")
        
        # Display file information
        if st.session_state.file_info:
            st.header("📋 File Information")
            info = st.session_state.file_info["summary"]
            st.info(f"""
            **File:** {st.session_state.file_info["filename"]}
            **Total Sheets:** {info["total_sheets"]}
            **Total Rows:** {info["total_rows"]:,}
            **Total Columns:** {info["total_columns"]}
            **Memory Usage:** {info["total_memory_usage"]/1024:.1f} KB
            """)
            
            # Display sheet details
            st.subheader("📄 Sheet Details")
            for sheet_name, sheet_info in info["sheets_info"].items():
                with st.expander(f"Sheet: {sheet_name}"):
                    st.write(f"**Rows:** {sheet_info['rows']:,}")
                    st.write(f"**Columns:** {sheet_info['columns']}")
                    st.write(f"**Memory:** {sheet_info['memory_usage']/1024:.1f} KB")
                    
                    st.write("**Column Details:**")
                    for col, dtype in sheet_info["data_types"].items():
                        missing = sheet_info["missing_values"][col]
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
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your data or general questions...")
        
        if user_input:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
            })
            
            # Process the query
            with st.spinner("Thinking..."):
                if st.session_state.dataframes is not None and chatbot.is_data_related_query(user_input):
                    # Data-related query
                    response = chatbot.chat_with_data(st.session_state.dataframes, user_input)
                else:
                    # General query
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
        sample_queries = [
            "What's the weather like today?",
            "Explain machine learning",
            "Show me all sheet names",
            "What is the average Occupancy On Books This Year from 1st Jan 2025 to 1st May",
            "what are total and unique market segments? Give me a list of all market segments",
            "Show statistics for all sheets"
        ]
        
        for query in sample_queries:
            if st.button(f"'{query}'", key=f"sample_{query}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                })
                
                if st.session_state.dataframes is not None and chatbot.is_data_related_query(query):
                    response = chatbot.chat_with_data(st.session_state.dataframes, query)
                else:
                    response = chatbot.general_chat(query)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
                st.rerun()

if __name__ == "__main__":
    main()