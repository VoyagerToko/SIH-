# main.py
import pandas as pd
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import textwrap
import os
import uuid
import json
import hashlib
import pickle
import time
import logging
import chromadb
from chromadb.utils import embedding_functions
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from logger_utils import log_function_call, log_process_start, log_process_end, log_error, log_file_operation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("eConsult")

# Step 1: Extract text from PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Step 2: Chunk text for retrieval
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Step 3: Build ChromaDB index of PDF
def build_index(chunks):
    # Use SentenceTransformerEmbeddingFunction from chromadb
    embed_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = chromadb.Client()

    # Try to get existing collection or create a new one
    try:
        collection = client.get_collection(name="pdf_chunks", embedding_function=embed_model)
    except Exception as e:
        logger.error(f"Failed to get collection, creating new one: {str(e)}")
        collection = client.create_collection(name="pdf_chunks", embedding_function=embed_model)
    
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )
    return collection, chunks, embed_model

# Step 4: Retrieve context
def retrieve_context(query, collection, chunks, embed_model, top_k=2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]

# Step 5: Ollama sentiment + summary
def classify_sentiment(text, context=""):
    prompt = f"""
    Analyze the following stakeholder comment in the context of the draft legislation.
    
    Categories:
    - Supportive: Comment expresses agreement, approval, or positive feedback about the proposal. Look for praise, agreement, or expressions that the proposal will have positive effects.
    - Critical: Comment expresses disagreement, disapproval, or negative feedback about the proposal. Look for concerns, objections, or statements about negative effects.
    - Neutral: Comment is factual, balanced, or doesn't express a clear stance on the proposal. Look for objective statements, questions seeking information, or balanced views.
    - Suggestive: Comment offers recommendations, alternative approaches, or constructive ideas. Look for phrases like "consider," "suggest," "could," "would be better if," etc.
    
    IMPORTANT GUIDELINES FOR BALANCED CLASSIFICATION:
    1. If a comment is clearly positive or approving, classify as Supportive.
    2. If a comment points out flaws, expresses concerns, or disapproval, classify as Critical.
    3. If a comment mainly offers suggestions, recommendations, or alternative approaches, classify as Suggestive.
    4. If a comment is factual, asks questions without expressing opinions, or presents balanced views, classify as Neutral.
    
    You MUST choose the ONE category that best matches the overall tone and content. DO NOT default to only using certain categories.
    
    Context: {context}
    Comment: "{text}"
    
    First analyze the sentiment thoroughly, focusing on the tone and specific language used, then respond with ONLY ONE of these exact category names: Supportive, Critical, Neutral, or Suggestive.
    Do not include any other text in your response.
    """
    logger.info(f"Classifying sentiment for comment: {text[:50]}...")
    
    resp = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    response = resp["message"]["content"].strip()
    
    valid_categories = ["Supportive", "Critical", "Neutral", "Suggestive"]
    for category in valid_categories:
        if category.lower() in response.lower():
            logger.info(f"Sentiment classified as: {category}")
            return category
    
    response_lower = response.lower()
    supportive_words = ["support", "positive", "agree", "good", "excellent", "approve", "like", "appreciate", "welcome"]
    critical_words = ["critic", "negative", "disagree", "bad", "poor", "concern", "issue", "problem", "oppose", "against", "flawed"]
    suggestive_words = ["suggest", "recommend", "could", "would", "may", "might", "consider", "propose", "alternative", "option", "improve"]
    neutral_words = ["neutral", "fact", "inform", "data", "balance", "question", "clarify", "ask", "what", "how", "when", "explain"]
    
    supportive_count = sum(1 for word in supportive_words if word in response_lower)
    critical_count = sum(1 for word in critical_words if word in response_lower)
    suggestive_count = sum(1 for word in suggestive_words if word in response_lower)
    neutral_count = sum(1 for word in neutral_words if word in response_lower)
    
    max_count = max(supportive_count, critical_count, suggestive_count, neutral_count)
    
    if max_count == 0:
        text_lower = text.lower()
        supportive_text_count = sum(1 for word in supportive_words if word in text_lower)
        critical_text_count = sum(1 for word in critical_words if word in text_lower)
        suggestive_text_count = sum(1 for word in suggestive_words if word in text_lower)
        neutral_text_count = sum(1 for word in neutral_words if word in text_lower)
        
        max_text_count = max(supportive_text_count, critical_text_count, suggestive_text_count, neutral_text_count)
        
        if max_text_count > 0:
            if supportive_text_count == max_text_count:
                logger.info("Sentiment classified as: Supportive (via text heuristics)")
                return "Supportive"
            elif critical_text_count == max_text_count:
                logger.info("Sentiment classified as: Critical (via text heuristics)")
                return "Critical"
            elif suggestive_text_count == max_text_count:
                logger.info("Sentiment classified as: Suggestive (via text heuristics)")
                return "Suggestive"
            else:
                logger.info("Sentiment classified as: Neutral (via text heuristics)")
                return "Neutral"
        
        logger.info("Sentiment classified as: Neutral (default)")
        return "Neutral"
    
    if supportive_count == max_count:
        logger.info("Sentiment classified as: Supportive (via heuristics)")
        return "Supportive"
    elif critical_count == max_count:
        logger.info("Sentiment classified as: Critical (via heuristics)")
        return "Critical"
    elif suggestive_count == max_count:
        logger.info("Sentiment classified as: Suggestive (via heuristics)")
        return "Suggestive"
    else:
        logger.info("Sentiment classified as: Neutral (via heuristics)")
        return "Neutral"

def summarize_comment(text, context=""):
    prompt = f"""
    Summarize the following stakeholder comment in one short, precise sentence.
    
    Context: {context}
    Comment: "{text}"
    
    Summary:
    """
    resp = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()

def generate_overall_summary(comments_list, sentiments):
    comment_samples = []
    for i, comment in enumerate(comments_list):
        if i < 15:
            text = comment["comment_text"][:150] + "..." if len(comment["comment_text"]) > 150 else comment["comment_text"]
            sentiment = comment["sentiment"]
            comment_samples.append(f"Comment {i+1} ({sentiment}): {text}")
    
    sentiment_counts = [f"{sentiment}: {count}" for sentiment, count in sentiments.items()]
    
    prompt = f"""
    Analyze the following stakeholder comments and their sentiment distribution to create a comprehensive summary.
    
    Sentiment Distribution:
    {', '.join(sentiment_counts)}
    
    Comment Samples:
    {' '.join(comment_samples)}
    
    Please provide a 3-4 paragraph summary that:
    1. Identifies the main themes and concerns across all comments
    2. Highlights the most significant sentiments and what they indicate
    3. Notes any important suggestions or feedback trends
    4. Provides a balanced conclusion about the overall stakeholder response
    
    Focus on concrete patterns rather than generalizations. Be specific about the key points raised in the comments.
    """
    
    resp = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()

# Step 5: Process comments
@log_function_call
def analyze_comments(pdf_path, comments_csv):
    logger.info(f"Loading PDF from: {pdf_path}")
    pdf_text = extract_pdf_text(pdf_path)
    logger.info(f"Extracted {len(pdf_text)} characters of text from PDF")
    
    chunk_time = log_process_start("Text Chunking")
    chunks = chunk_text(pdf_text)
    log_process_end("Text Chunking", chunk_time, {"chunks_created": len(chunks)})
    
    index_time = log_process_start("Building Search Index")
    collection, chunks, embed_model = build_index(chunks)
    log_process_end("Building Search Index", index_time)

    logger.info(f"Loading comments from: {comments_csv}")
    comments = pd.read_csv(comments_csv)
    logger.info(f"Loaded CSV with {len(comments)} rows and {len(comments.columns)} columns")
    logger.info(f"Available columns: {', '.join(comments.columns.tolist())}")
    
    if "Index" in comments.columns and "index" not in comments.columns:
        comments["index"] = comments["Index"]
        logger.info("Using 'Index' column from CSV")
    elif "index" not in comments.columns:
        comments["index"] = [f"Q{i+1}" for i in range(len(comments))]
        logger.warning("No index column found, created default indices")
        
    if "Comments" in comments.columns and "comments" not in comments.columns:
        comments["comments"] = comments["Comments"]
        logger.info("Using 'Comments' column from CSV")
    elif "comments" not in comments.columns:
        alternative_cols = ["comment_text", "Comment", "Feedback", "Response", "Suggestion"]
        for col in alternative_cols:
            if col in comments.columns:
                comments["comments"] = comments[col]
                logger.info(f"Using '{col}' as comments column")
                break
        else:
            error_msg = f"No suitable comments column found. Please ensure your CSV contains one of these: 'comments', {', '.join(alternative_cols)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    logger.info(f"Processing {len(comments)} comments for analysis")

    analysis_time = log_process_start("Comment Analysis")
    results = []
    
    for idx, (_, row) in enumerate(comments.iterrows()):
        cid = row["index"]
        text = row["comments"]
        
        logger.info(f"Processing comment {idx+1}/{len(comments)}: ID={cid}")
        truncated = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"Comment text (truncated): {truncated}")

        context_time = time.time()
        context = " ".join(retrieve_context(text, collection, chunks, embed_model, top_k=2))
        logger.info(f"Context retrieval took {time.time() - context_time:.2f}s")
        
        sentiment_time = time.time()
        sentiment = classify_sentiment(text, context)
        logger.info(f"Sentiment classification took {time.time() - sentiment_time:.2f}s, Result: {sentiment}")
        
        summary_time = time.time()
        summary = summarize_comment(text, context)
        logger.info(f"Comment summarization took {time.time() - summary_time:.2f}s")

        results.append({
            "comment_id": cid,
            "comment_text": text,
            "sentiment": sentiment,
            "summary": summary,
            "context_used": context[:200] + "..."
        })
    
    sentiment_distribution = {
        sentiment: len([r for r in results if r["sentiment"] == sentiment])
        for sentiment in set(r["sentiment"] for r in results)
    }
    
    logger.info("Generating overall comment summary")
    overall_summary_time = time.time()
    overall_summary = generate_overall_summary(results, sentiment_distribution)
    logger.info(f"Overall summary generation took {time.time() - overall_summary_time:.2f}s")
    
    summary_row = {
        "comment_id": "SUMMARY",
        "comment_text": f"Total Comments: {len(comments)}",
        "sentiment": "Summary",
        "summary": overall_summary,
        "context_used": f"Sentiment Distribution: {', '.join([f'{key}: {value}' for key, value in sentiment_distribution.items()])}"
    }
    
    results.append(summary_row)
    
    log_process_end("Comment Analysis", analysis_time, {
        "comments_processed": len(comments),
        "sentiment_distribution": sentiment_distribution
    })

    df = pd.DataFrame(results)
    logger.info(f"Analysis complete. Results DataFrame created with {len(df)} rows")
    return df, comments

# Step 6: Word cloud
def make_wordcloud(comments):
    all_text = " ".join(comments["comment_text"].astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(all_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("wordcloud.png")
    print("✅ Wordcloud saved as wordcloud.png")

# Flask Application Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'csv'}

@app.route('/', methods=['GET'])
@log_function_call
def index():
    logger.info("Rendering index page")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@log_function_call
def analyze():
    logger.info("Received analysis request")
    upload_process_time = log_process_start("File Upload Process")
    
    if 'pdf_file' not in request.files or 'csv_file' not in request.files:
        log_error("Missing required files")
        flash('Both PDF and CSV files are required')
        return redirect(request.url)
    
    pdf_file = request.files['pdf_file']
    csv_file = request.files['csv_file']
    
    if pdf_file.filename == '' or csv_file.filename == '':
        log_error("Empty filename(s)")
        flash('No selected file')
        return redirect(request.url)
    
    if pdf_file and csv_file and allowed_file(pdf_file.filename) and allowed_file(csv_file.filename):
        pdf_filename = secure_filename(pdf_file.filename)
        csv_filename = secure_filename(csv_file.filename)
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        
        log_file_operation("Saving PDF file", pdf_path)
        pdf_file.save(pdf_path)
        
        log_file_operation("Saving CSV file", csv_path)
        csv_file.save(csv_path)
        
        log_process_end("File Upload Process", upload_process_time)
        
        try:
            analysis_process_time = log_process_start("Comment Analysis")
            results_df, comments = analyze_comments(pdf_path, csv_path)
            
            viz_process_time = log_process_start("Generating Visualizations")
            wordcloud_path = os.path.join('static', f'wordcloud_{uuid.uuid4()}.png')
            
            all_text = " ".join(results_df["comment_text"].astype(str).tolist())
            wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(all_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path)
            plt.close()
            
            sentiment_counts = results_df['sentiment'].value_counts().to_dict()
            
            response_data = {
                'success': True,
                'wordcloud_path': wordcloud_path,
                'sentiment_counts': sentiment_counts,
                'results': results_df.to_dict(orient='records')
            }
            
            log_process_end("Generating Visualizations", viz_process_time)
            log_process_end("Comment Analysis", analysis_process_time)
            
            return jsonify(response_data)
            
        except Exception as e:
            log_error(f"Analysis failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    log_error("Invalid file type")
    flash('Invalid file type. Only PDF and CSV files are allowed.')
    return redirect(request.url)

if __name__ == "__main__":
    logger.info("Starting eConsult Feedback Analysis Tool")
    app.run(debug=True, host='0.0.0.0', port=5000)