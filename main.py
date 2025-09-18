# econsult_prototype.py
import pandas as pd
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib
import os

####################################
# Performance Configuration
####################################
# Adjust these settings to optimize performance for your specific hardware
PARALLEL_WORKERS = min(os.cpu_count() or 4, 8)  # Max workers for parallel processing
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding calculation
CACHE_ENABLED = True  # Whether to use caching for LLM calls and index
LLM_CACHE_SIZE = 1024  # Maximum number of cached LLM responses
CONTEXT_CACHE_SIZE = 512  # Maximum number of cached context retrievals
INDEX_CACHE_DIR = "cache"  # Directory to store cached indices
####################################
# Set the backend to Agg to avoid GUI issues with Matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import textwrap
import uuid
import json
import hashlib
import pickle
import time
import logging
import sys
import concurrent.futures
import functools
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from functools import lru_cache

# Create cache directory if it doesn't exist
if not os.path.exists(INDEX_CACHE_DIR):
    os.makedirs(INDEX_CACHE_DIR, exist_ok=True)

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

# Import custom logger utilities
from werkzeug.utils import secure_filename
from logger_utils import log_function_call, log_process_start, log_process_end, log_error, log_file_operation

####################################
# Step 1: Extract text from PDF
####################################
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

####################################
# Step 2: Chunk text for retrieval
####################################
def chunk_text(text, chunk_size=500, overlap=50):
    """Chunk text more efficiently using numpy operations"""
    words = text.split()
    
    # Pre-allocate result size for better performance
    n_chunks = max(1, (len(words) - overlap) // (chunk_size - overlap))
    chunks = []
    
    # Use more efficient list operations with pre-calculated indices
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

####################################
# Step 3: Build FAISS index of PDF
####################################
def build_index(chunks, batch_size=EMBEDDING_BATCH_SIZE):
    """Build FAISS index with batched processing for better memory efficiency"""
    # Initialize model - use a smaller, faster model if accuracy can be slightly compromised
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Process in batches to reduce memory usage
    embeddings_list = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(batch_embeddings)
        embeddings_list.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(embeddings_list)
    
    # Create and optimize the index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    # Add all normalized embeddings at once
    index.add(embeddings)
    
    return index, chunks, embed_model

# Cache context retrieval results to avoid redundant embedding computations
@lru_cache(maxsize=CONTEXT_CACHE_SIZE)
def retrieve_context(query, index_id, top_k=2):
    """Retrieve context using the index, with caching for repeated queries"""
    # Get the global variables since we can't pass them directly to the cached function
    index = _indices.get(index_id, None)
    chunks = _chunks.get(index_id, None)
    embed_model = _embed_models.get(index_id, None)
    
    if not all([index, chunks, embed_model]):
        logger.error(f"Missing index components for ID {index_id}")
        return ["Context retrieval failed."]
    
    # Encode query
    emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    
    # Search index
    D, I = index.search(emb, top_k)
    
    # Return relevant chunks
    return [chunks[i] for i in I[0]]

# Global storage for indices, chunks, and models to be used with the cached function
_indices = {}
_chunks = {}
_embed_models = {}

# Function to register an index for use with the cached retrieve_context
def register_index(index, chunks, embed_model):
    """Register an index with a unique ID for use with the cached retrieve function"""
    index_id = str(uuid.uuid4())
    _indices[index_id] = index
    _chunks[index_id] = chunks
    _embed_models[index_id] = embed_model
    return index_id

####################################
# Step 4: Ollama sentiment + summary
####################################
# Create a LLM response cache with a maximum size
# Using lru_cache for in-memory caching of LLM calls
@lru_cache(maxsize=LLM_CACHE_SIZE)
def _cached_ollama_call(prompt_hash, model_name="mistral"):
    """Cache wrapper for ollama calls to avoid redundant API calls"""
    # Retrieve the prompt from the hash-to-prompt mapping
    prompt = _prompt_cache.get(prompt_hash)
    if not prompt:
        logger.warning(f"Cache miss for prompt hash {prompt_hash}")
        return None
        
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Dictionary to store prompts by their hash
_prompt_cache = {}

def _get_cached_llm_response(prompt, model_name="mistral"):
    """Get a cached LLM response or generate a new one"""
    # Create a hash of the prompt for caching
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    
    # Store the prompt in the mapping
    _prompt_cache[prompt_hash] = prompt
    
    try:
        # Try to get the cached response
        return _cached_ollama_call(prompt_hash, model_name)
    except Exception as e:
        logger.warning(f"Error retrieving from cache: {e}. Calling LLM directly.")
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()

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
    
    response = _get_cached_llm_response(prompt)
    
    # Ensure the response is one of the valid categories
    valid_categories = ["Supportive", "Critical", "Neutral", "Suggestive"]
    
    # Clean up response and handle edge cases
    for category in valid_categories:
        if category.lower() in response.lower():
            logger.info(f"Sentiment classified as: {category}")
            return category
    
    # If no valid category found, use improved heuristics to assign one
    response_lower = response.lower()
    
    # Look for more specific keywords to improve categorization
    supportive_words = ["support", "positive", "agree", "good", "excellent", "approve", "like", "appreciate", "welcome"]
    critical_words = ["critic", "negative", "disagree", "bad", "poor", "concern", "issue", "problem", "oppose", "against", "flawed"]
    suggestive_words = ["suggest", "recommend", "could", "would", "may", "might", "consider", "propose", "alternative", "option", "improve"]
    neutral_words = ["neutral", "fact", "inform", "data", "balance", "question", "clarify", "ask", "what", "how", "when", "explain"]
    
    # Count matches for each category
    supportive_count = sum(1 for word in supportive_words if word in response_lower)
    critical_count = sum(1 for word in critical_words if word in response_lower)
    suggestive_count = sum(1 for word in suggestive_words if word in response_lower)
    neutral_count = sum(1 for word in neutral_words if word in response_lower)
    
    # Determine the category with the most matches
    max_count = max(supportive_count, critical_count, suggestive_count, neutral_count)
    
    if max_count == 0:
        # If no matches, check the original text for sentiment indicators
        text_lower = text.lower()
        supportive_text_count = sum(1 for word in supportive_words if word in text_lower)
        critical_text_count = sum(1 for word in critical_words if word in text_lower)
        suggestive_text_count = sum(1 for word in suggestive_words if word in text_lower)
        neutral_text_count = sum(1 for word in neutral_words if word in text_lower)
        
        # Find the dominant sentiment in the original text
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
        
        # Default if nothing matches
        logger.info("Sentiment classified as: Neutral (default)")
        return "Neutral"
    
    # Return the category with the most matches
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
    return _get_cached_llm_response(prompt)

def generate_overall_summary(comments_list, sentiments):
    """Generate a comprehensive summary of all comments and their sentiment trends."""
    # Prepare a condensed representation of comments and their sentiments
    comment_samples = []
    for i, comment in enumerate(comments_list):
        # Take a sample of comments (up to 15)
        if i < 15:
            # Get comment text (truncated if needed)
            text = comment["comment_text"][:150] + "..." if len(comment["comment_text"]) > 150 else comment["comment_text"]
            sentiment = comment["sentiment"]
            comment_samples.append(f"Comment {i+1} ({sentiment}): {text}")
    
    # Format the sentiment distribution
    sentiment_counts = []
    for sentiment, count in sentiments.items():
        sentiment_counts.append(f"{sentiment}: {count}")
    
    # Create the prompt for overall summary
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
    
    return _get_cached_llm_response(prompt)

####################################
# Step 5: Process comments
####################################
@log_function_call
def analyze_comments(pdf_path, comments_csv):
    # 1. Load PDF & build retriever
    logger.info(f"Loading PDF from: {pdf_path}")
    pdf_text = extract_pdf_text(pdf_path)
    logger.info(f"Extracted {len(pdf_text)} characters of text from PDF")
    
    # Calculate PDF hash for caching
    pdf_hash = hashlib.md5(pdf_text.encode()).hexdigest()[:10]
    cache_file = os.path.join(INDEX_CACHE_DIR, f"index_cache_{pdf_hash}.pkl")
    
    # Try to load from cache first
    index_id = None
    if os.path.exists(cache_file):
        try:
            logger.info(f"Loading index from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                index_id = cached_data['index_id']
                # Check if this index_id exists in our registry
                if index_id in _indices:
                    logger.info(f"Using cached index with ID {index_id}")
                else:
                    # Register the cached components
                    _indices[index_id] = cached_data['index']
                    _chunks[index_id] = cached_data['chunks']
                    _embed_models[index_id] = cached_data['embed_model']
                    logger.info(f"Registered cached index with ID {index_id}")
        except Exception as e:
            logger.error(f"Error loading cached index: {e}")
            index_id = None
    
    # If not cached, build the index
    if not index_id:
        chunk_time = log_process_start("Text Chunking")
        chunks = chunk_text(pdf_text)
        log_process_end("Text Chunking", chunk_time, {"chunks_created": len(chunks)})
        
        index_time = log_process_start("Building Search Index")
        index, chunks, embed_model = build_index(chunks)
        index_id = register_index(index, chunks, embed_model)
        log_process_end("Building Search Index", index_time)
        
        # Cache the index for future use
        try:
            logger.info(f"Saving index to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'index_id': index_id,
                    'index': index,
                    'chunks': chunks,
                    'embed_model': embed_model
                }, f)
        except Exception as e:
            logger.error(f"Error caching index: {e}")

    # 2. Load comments with flexible column handling
    logger.info(f"Loading comments from: {comments_csv}")
    comments = pd.read_csv(comments_csv)
    logger.info(f"Loaded CSV with {len(comments)} rows and {len(comments.columns)} columns")
    logger.info(f"Available columns: {', '.join(comments.columns.tolist())}")
    
    # Check for required columns and handle alternatives
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
        # Try to find an alternative column
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

    # Define a worker function to process a single comment
    def process_comment(idx_row):
        idx, (_, row) = idx_row
        cid = row["index"]
        text = row["comments"]
        
        logger.info(f"Processing comment {idx+1}/{len(comments)}: ID={cid}")
        
        # Log truncated comment text (for privacy)
        truncated = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"Comment text (truncated): {truncated}")

        # Retrieve context using the cached function
        context_time = time.time()
        context = " ".join(retrieve_context(text, index_id, top_k=2))
        logger.info(f"Context retrieval took {time.time() - context_time:.2f}s")
        
        # Classify sentiment
        sentiment_time = time.time()
        sentiment = classify_sentiment(text, context)
        logger.info(f"Sentiment classification took {time.time() - sentiment_time:.2f}s, Result: {sentiment}")
        
        # Generate summary
        summary_time = time.time()
        summary = summarize_comment(text, context)
        logger.info(f"Comment summarization took {time.time() - summary_time:.2f}s")

        return {
            "comment_id": cid,
            "comment_text": text,
            "sentiment": sentiment,
            "summary": summary,
            "context_used": context[:200] + "..."
        }

    analysis_time = log_process_start("Comment Analysis")
    
    # Process comments in parallel using the configured number of workers
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # Submit all comments for processing
        future_to_idx = {executor.submit(process_comment, item): item for item in enumerate(comments.iterrows())}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed comment {idx[0]+1}/{len(comments)}")
            except Exception as exc:
                logger.error(f"Comment {idx[0]+1} generated an exception: {exc}")
                # Continue with other comments even if one fails
    
    # Calculate sentiment statistics
    sentiment_distribution = {
        sentiment: len([r for r in results if r["sentiment"] == sentiment])
        for sentiment in set(r["sentiment"] for r in results)
    }
    
    # Generate an overall summary of all comments
    logger.info("Generating overall comment summary")
    overall_summary_time = time.time()
    overall_summary = generate_overall_summary(results, sentiment_distribution)
    logger.info(f"Overall summary generation took {time.time() - overall_summary_time:.2f}s")
    
    # Add a summary row with total comments, sentiment distribution, and overall summary
    summary_row = {
        "comment_id": "SUMMARY",
        "comment_text": f"Total Comments: {len(comments)}",
        "sentiment": "Summary",
        "summary": overall_summary,
        "context_used": f"Sentiment Distribution: {', '.join([f'{key}: {value}' for key, value in sentiment_distribution.items()])}"
    }
    
    # Add summary row to results
    results.append(summary_row)
    
    log_process_end("Comment Analysis", analysis_time, {
        "comments_processed": len(comments),
        "sentiment_distribution": sentiment_distribution
    })

    df = pd.DataFrame(results)
    logger.info(f"Analysis complete. Results DataFrame created with {len(df)} rows")
    return df, comments

####################################
# Step 6: Word cloud
####################################
def make_wordcloud(comments):
    all_text = " ".join(comments["comment_text"].astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(all_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("wordcloud.png")
    print("✅ Wordcloud saved as wordcloud.png")

####################################
# Flask Application Setup
####################################
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Create uploads folder if it doesn't exist
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
    
    # Check if files are selected
    if pdf_file.filename == '' or csv_file.filename == '':
        log_error("Empty filename(s)")
        flash('No selected file')
        return redirect(request.url)
    
    if pdf_file and csv_file and allowed_file(pdf_file.filename) and allowed_file(csv_file.filename):
        # Save the files
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
            # Run the analysis
            results_df, comments = analyze_comments(pdf_path, csv_path)
            
            # Generate wordcloud
            viz_process_time = log_process_start("Generating Visualizations")
            wordcloud_path = os.path.join('static', f'wordcloud_{uuid.uuid4()}.png')
            
            # Generate wordcloud
            all_text = " ".join(results_df["comment_text"].astype(str).tolist())
            wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(all_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path)
            plt.close()
            
            # Count sentiments
            sentiment_counts = results_df['sentiment'].value_counts().to_dict()
            
            # Prepare data for the response
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

####################################
# MAIN
####################################
if __name__ == "__main__":
    logger.info("Starting eConsult Feedback Analysis Tool")
    app.run(debug=True, host='0.0.0.0', port=5000)
