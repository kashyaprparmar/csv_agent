# 📊 Multi-Sheet Excel Chat Assistant with AI Judge

A Streamlit application that allows you to upload multiple Excel files and chat with your data using natural language queries. The application leverages Google's Gemini 2.5 Flash, PandasAI, and includes an innovative AI Judge system for response verification.

## ✨ Features

### 🔍 Core Functionality
- **Multi-File Support**: Upload and analyze multiple Excel files simultaneously
- **Multi-Sheet Processing**: Automatically processes all sheets within each Excel file
- **Natural Language Queries**: Ask questions about your data in plain English
- **Intelligent Query Routing**: Automatically determines whether queries are data-related or general

### 🤖 AI-Powered Analysis
- **Google Gemini Integration**: Uses Gemini 2.5 Flash for advanced language understanding
- **PandasAI Agent**: Specialized data analysis with conversation memory
- **AI Judge System**: Automatic verification of response quality and accuracy
- **Smart Fallback**: Graceful handling when primary analysis fails

### 📈 Data Visualization & Analysis
- **Interactive DataFrames**: View results in formatted tables
- **Statistical Analysis**: Perform complex calculations and aggregations
- **Data Summaries**: Automatic generation of data structure insights
- **Memory Optimization**: Efficient handling of large datasets

### 🎨 User Interface
- **Modern Design**: Clean, responsive Streamlit interface
- **Real-time Chat**: Interactive chat interface with message history
- **File Information Panel**: Detailed overview of uploaded files and sheets
- **Sample Queries**: Pre-built example queries for quick testing
- **Judge Verdict Display**: Visual feedback on response quality

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- Google API Key (from [AI Studio](https://aistudio.google.com/))

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd excel-chat-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google API Key**
   
   Choose one of these methods:
   
   **Option A: Environment Variable**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   
   **Option B: Streamlit Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```
   
   **Option C: In-App Input**
   Enter your API key directly in the sidebar when prompted

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 🚀 Usage

### Getting Started

1. **Launch the Application**
   - Run the Streamlit app and open it in your browser
   - Enter your Google API Key if not already configured

2. **Upload Excel Files**
   - Use the sidebar file uploader to select one or more Excel files (.xlsx, .xls)
   - The app will automatically process all sheets in each file
   - Wait for the AI agent to initialize (shown in sidebar status)

3. **Start Chatting**
   - Type your questions in the chat input at the bottom
   - Use either data-specific queries or general questions
   - View responses with AI judge verification

#### Example Data Analysis Queries
```
- "What is my budget per product in 2025?"
- "Show me the average sales for each month"
- "What are the best selling products?"
- "Compare revenue between this year and last year"
- "What is the correlation between price and quantity?"
- "Show me the top 5 customers by sales"
- "Filter data where sales is greater than 1000"
```


## 🏗️ Architecture

### Core Components

1. **ExcelChatBot Class**
   - Main orchestrator for all functionality
   - Handles model initialization and API management
   - Routes queries between data analysis and general chat

2. **Multi-Model System**
   - **Chat Model**: Gemini 2.5 Flash for general conversation
   - **Judge Model**: Lower temperature Gemini for consistent evaluation
   - **PandasAI Agent**: Specialized data analysis with conversation memory

3. **AI Judge System**
   - Evaluates response accuracy and completeness
   - Provides confidence scores and detailed feedback
   - Offers suggestions for improvement

### Data Processing Pipeline

```
Excel Files → Multi-Sheet Extraction → DataFrame Creation → 
Agent Initialization → Query Processing → AI Judge Verification → 
Response Display
```

## 📋 Key Features Explained

### AI Judge System

The AI Judge is a unique feature that automatically evaluates every data-related response based on:

- **Accuracy**: Correctness of the analysis
- **Completeness**: Sufficiency of information provided
- **Data Usage**: Appropriate use of available data
- **Methodology**: Soundness of analytical approach
- **Clarity**: Understandability of the response

Each response receives:
- ✅ **APPROVED**: High-quality, accurate response
- ❌ **REJECTED**: Issues found, needs review
- ⚠️ **ERROR**: System error occurred
- ❓ **UNKNOWN**: Unable to determine quality

### Smart Query Routing

The system intelligently determines query types:
- **Data-related**: Routed to PandasAI agent with judge verification
- **General**: Handled by Gemini chat model
- **Hybrid**: Context-aware processing

### Memory & Performance

- **Conversation Memory**: Agent remembers previous interactions (6 message pairs)
- **Efficient Loading**: Optimized DataFrame processing
- **Memory Monitoring**: Real-time memory usage tracking
- **Error Handling**: Graceful fallbacks for all failure scenarios

## 🔧 Configuration

### Environment Variables

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### Streamlit Configuration

The app includes custom CSS styling and responsive design. You can modify the appearance by editing the CSS in the `st.markdown()` sections.

### Model Configuration

- **Chat Model**: `gemini-2.5-flash-preview-05-20` (Temperature: 0.7)
- **Judge Model**: Same model with lower temperature (0.1) for consistency
- **Agent Memory**: 6 conversation pairs retained

## 🐛 Troubleshooting

### Common Issues

**API Key Not Working**
- Verify your API key is valid and has proper permissions
- Check that the key is correctly set in environment or secrets

**Agent Loading Fails**
- Ensure your Excel files are properly formatted
- Check for corrupted or empty sheets
- Verify sufficient system memory

**Poor Response Quality**
- Try rephrasing your query more specifically
- Check the AI Judge feedback for improvement suggestions
- Ensure your data contains the information you're asking about

**Performance Issues**
- Large files may take longer to process
- Consider breaking down complex queries
- Monitor memory usage in the sidebar

### Error Messages

- **"AI agent is not available"**: Wait for agent initialization to complete
- **"Parsing error"**: Try rephrasing your query or check data format
- **"System error"**: Check your API key and internet connection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies in development mode
4. Make your changes
5. Test thoroughly
6. Submit a pull request

## Acknowledgments

- **Google Gemini AI** for powerful language understanding
- **PandasAI** for intelligent data analysis capabilities
- **Streamlit** for the excellent web app framework
- **LangChain** for AI model orchestration

## 📞 Support

For support, issues, or feature requests, please:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Include error messages and steps to reproduce

---

**Built with ❤️ for data analysts, business users, and anyone who wants to chat with their Excel data!**