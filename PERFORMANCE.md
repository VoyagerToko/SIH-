# Performance Optimizations

This document outlines the performance optimizations implemented in the Consultation Feedback Analysis Tool.

## Key Optimizations

### 1. Combined LLM Calls

- **Benefit**: Reduces API calls by 50% for comment analysis
- **Implementation**: The `analyze_and_summarize_comment` function performs both sentiment analysis and summarization in a single LLM call
- **How it works**: A structured prompt asks the model to produce both sentiment classification and summary in a specific format

### 2. Parallel Processing

- **Benefit**: Significantly speeds up processing for large comment datasets
- **Implementation**: Uses `concurrent.futures.ThreadPoolExecutor` to process multiple comments simultaneously
- **Configuration**: Number of workers is configurable via `PARALLEL_WORKERS` constant (defaults to CPU count)

### 3. Caching System

- **In-Memory Caching**:
  - LRU cache for LLM responses to avoid redundant API calls
  - Context retrieval results cached to avoid duplicate embeddings
  
- **Disk Caching**:
  - FAISS indices cached to disk to avoid rebuilding for the same PDF
  - Cache directory configurable via `INDEX_CACHE_DIR`

### 4. Optimized Embedding & Retrieval

- Batch processing for embedding calculations
- Pre-computed embeddings for efficient retrieval
- Normalized vectors for better search performance

### 5. Efficient Text Chunking

- Optimized chunk generation algorithm
- Pre-allocated memory for better performance

## Configuration Options

Performance-related configuration can be adjusted at the top of `main.py`:

```python
# Performance Configuration
PARALLEL_WORKERS = min(os.cpu_count() or 4, 8)  # Max workers for parallel processing
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding calculation
CACHE_ENABLED = True  # Whether to use caching for LLM calls and index
LLM_CACHE_SIZE = 1024  # Maximum number of cached LLM responses
CONTEXT_CACHE_SIZE = 512  # Maximum number of cached context retrievals
INDEX_CACHE_DIR = "cache"  # Directory to store cached indices
```

## Future Optimization Opportunities

1. **Batched LLM Processing**: Group similar sentiment classification or summarization tasks
2. **Faster Embedding Models**: Consider smaller/quantized models for speed
3. **Database Caching**: Move to persistent database for production scale
4. **Distributed Processing**: Scale across multiple servers for large datasets
5. **Real-time Progress Reporting**: Implement WebSockets for streaming results