# Consultation Feedback Analysis Tool

This is a Flask web application for analyzing stakeholder feedback on consultation documents. It uses NLP techniques to classify sentiment, summarize comments, and generate visualizations.

## Features

- Upload consultation documents (PDF) and stakeholder comments (CSV)
- AI-powered sentiment analysis using Ollama and Mistral
- Contextual retrieval using FAISS vector database
- Comment summarization
- Data visualization with Chart.js and Wordcloud
- Comprehensive logging system

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama and download the Mistral model:
   ```bash
   ollama pull mistral
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Open your browser and navigate to http://localhost:5000
3. Upload your PDF document and CSV file with stakeholder comments
4. View the analysis results

## Troubleshooting

### Method Not Allowed Error

If you encounter a "Method Not Allowed" error when submitting the form:

1. Check that the form field names match between HTML and Python:
   - The HTML form should use `name="pdf_file"` and `name="csv_file"` 
   - The Python code should look for `request.files['pdf_file']` and `request.files['csv_file']`

2. Ensure proper form submission method:
   - The form should have `method="POST"` and `enctype="multipart/form-data"` attributes
   - The route should be decorated with `@app.route('/analyze', methods=['POST'])`

### TypeError in log_file_operation

If you encounter `TypeError: log_file_operation() missing 1 required positional argument: 'file_path'`:

1. Make sure to call the function with both required parameters:
   ```python
   log_file_operation("Operation description", file_path)
   ```

2. The first parameter should be an operation description (string), and the second parameter should be the file path.

### Missing start_time in log_process_end

If you use the process logging functions, remember to:

1. Store the return value from `log_process_start`:
   ```python
   process_time = log_process_start("Process Name")
   ```

2. Pass this value to `log_process_end`:
   ```python
   log_process_end("Process Name", process_time)
   ```

## CSV Format

The application supports CSVs with the following column structure:
- Must have either "Index" or "index" column for comment IDs
- Must have either "Comments" or "comments" column for the comment text
- Alternative column names for comments: "comment_text", "Comment", "Feedback", "Response", "Suggestion"

## Logging

The application includes comprehensive logging throughout the codebase:
- Function calls with execution times
- File operations
- Process tracking
- Error handling

Log files are stored in `app.log`.