# 📊 Multi-Sheet Excel Chat Assistant

A powerful Streamlit application that allows you to chat with your Excel files using AI. Upload your Excel file and ask questions about your data in natural language!

## ✨ Features

- **Multi-Sheet Support**: Works with Excel files containing multiple worksheets
- **Natural Language Queries**: Ask questions about your data in plain English
- **Data Analysis**: Get insights, statistics, and visualizations from your data
- **General Chat**: Also works as a general-purpose AI assistant
- **Real-time Processing**: Instant responses to your queries
- **Interactive UI**: Clean and user-friendly interface

## 🚀 Getting Started

### Prerequisites

- Python >=3.10 and <=3.12
- Google API Key (Gemini AI)

### Installation

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install streamlit pandas langchain-google-genai pandasai openpyxl
   ```

3. **Get your Google API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Create an account and generate an API key
   - Keep this key handy - you'll need it to run the app

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Enter your Google API Key** in the sidebar when prompted

4. **Upload your Excel file** using the file uploader

5. **Start chatting** with your data!

## 📝 How to Use

### Uploading Files
- Click on "Upload your Excel file" in the sidebar
- Select your `.xlsx` or `.xls` file
- The app will automatically load all sheets and display file information

### Asking Questions

**Data-related queries** (when you have a file uploaded):
- "Show me the first 10 rows of data"
- "What are the column names in Sheet1?"
- "Calculate the average of the Sales column"
- "Show me statistics for all sheets"
- "Plot a chart of Revenue by Month"

**General queries**:
- "What's the weather like today?"
- "Explain machine learning"
- "Write a Python function to sort a list"

### Sample Data Queries

The app includes sample query buttons to help you get started:
- Show sheet information
- Get data statistics
- Analyze specific columns
- Filter and group data

## 🔧 Features Explained

### Smart Query Detection
The app automatically detects whether your question is about the uploaded data or a general inquiry, routing it to the appropriate AI model.

### Multi-Sheet Analysis
- View information about all sheets in your Excel file
- Ask questions across multiple sheets
- Get combined analysis results

### Data Visualization
- Generate charts and plots from your data
- View data in formatted tables
- Get statistical summaries

### Error Handling
- Fallback responses when data analysis fails
- Clear error messages
- Graceful handling of various file formats

## 📊 File Information Display

The sidebar shows detailed information about your uploaded file:
- Total number of sheets
- Row and column counts
- Memory usage
- Column data types
- Missing value counts

## 🛠️ Technical Details

- **Frontend**: Streamlit
- **AI Models**: Google Gemini (via LangChain)
- **Data Analysis**: PandasAI
- **File Processing**: Pandas, OpenPyXL
- **Styling**: Custom CSS for enhanced UI

## 🔒 Privacy & Security

- Your API key is stored only in the session
- Files are processed locally and temporarily
- No data is permanently stored
- Temporary files are automatically cleaned up

## 📋 Requirements

```
streamlit
pandas
langchain-google-genai
pandasai
openpyxl
```

## 🐛 Troubleshooting

**Common Issues:**

1. **API Key Error**: Make sure you've entered a valid Google API key
2. **File Upload Error**: Ensure your Excel file is not corrupted and has readable sheets
3. **Query Not Working**: Try rephrasing your question or use the sample queries
4. **Slow Response**: Large files may take longer to process

**Getting Help:**
- Check the error messages in the app
- Try sample queries to test functionality
- Ensure your Excel file has proper data formatting

## 🚀 Tips for Best Results

1. **Clear Questions**: Be specific about what you want to analyze
2. **Column Names**: Use exact column names from your data
3. **Sheet References**: Mention specific sheet names when needed
4. **Data Types**: Ensure your data is properly formatted in Excel

## 📈 Example Use Cases

- **Business Analytics**: Analyze sales data, customer information, financial reports
- **Research Data**: Process survey results, experimental data, statistical analysis
- **Personal Finance**: Track expenses, budget analysis, investment tracking
- **Project Management**: Analyze project timelines, resource allocation, progress tracking

---

**Note**: This application requires an active internet connection to communicate with Google's Gemini AI service.