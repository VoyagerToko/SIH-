"""
Cache module for storing and retrieving embeddings.
"""
import os
import pickle
import hashlib
import time
import logging

# Get logger from main
logger = logging.getLogger("eConsult")

class EmbeddingsCache:
    """
    A class for caching embeddings to avoid regenerating them for the same documents.
    """
    
    def __init__(self, cache_dir="cache"):
        """
        Initialize the cache with the specified directory.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")
        else:
            logger.info(f"Using existing cache directory: {cache_dir}")

    def _get_cache_key(self, file_path):
        """
        Generate a unique cache key for a file based on its content hash and path.
        
        Args:
            file_path: Path to the file to generate a key for
            
        Returns:
            A unique hash string
        """
        try:
            # Use file content hash + modification time for cache key
            file_hash = hashlib.md5()
            
            # Add file path to hash to make it unique for different files with the same content
            file_hash.update(os.path.abspath(file_path).encode('utf-8'))
            
            # Add file modification time to hash to invalidate cache when file changes
            file_hash.update(str(os.path.getmtime(file_path)).encode('utf-8'))
            
            # Add file size to hash
            file_hash.update(str(os.path.getsize(file_path)).encode('utf-8'))
            
            # Calculate hash from file content (first 1MB for large files)
            with open(file_path, 'rb') as f:
                # Read up to 1MB of the file for hashing
                file_hash.update(f.read(1024 * 1024))
                
            return file_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Error generating cache key for {file_path}: {str(e)}")
            # Fall back to a basic key
            return hashlib.md5(os.path.abspath(file_path).encode('utf-8')).hexdigest()

    def get_cache_path(self, file_path):
        """
        Get the cache file path for the given file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Path to the cache file
        """
        cache_key = self._get_cache_key(file_path)
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get_cached_embeddings(self, file_path):
        """
        Retrieve cached embeddings for the given file if they exist.
        
        Args:
            file_path: Original file path
            
        Returns:
            Cached data or None if no cache exists
        """
        cache_path = self.get_cache_path(file_path)
        
        if os.path.exists(cache_path):
            try:
                logger.info(f"Cache found for {os.path.basename(file_path)}")
                start_time = time.time()
                
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                logger.info(f"Cache loaded in {time.time() - start_time:.2f}s")
                return cached_data
            except Exception as e:
                logger.warning(f"Error loading cache for {os.path.basename(file_path)}: {str(e)}")
                # If there's an error loading cache, return None to regenerate it
                return None
        
        logger.info(f"No cache found for {os.path.basename(file_path)}")
        return None

    def cache_embeddings(self, file_path, data):
        """
        Cache embeddings for the given file.
        
        Args:
            file_path: Original file path
            data: Data to cache
            
        Returns:
            True if caching was successful, False otherwise
        """
        cache_path = self.get_cache_path(file_path)
        
        try:
            logger.info(f"Caching embeddings for {os.path.basename(file_path)}")
            start_time = time.time()
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            cache_size = os.path.getsize(cache_path)
            logger.info(f"Cache saved in {time.time() - start_time:.2f}s ({cache_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            logger.error(f"Error caching embeddings for {os.path.basename(file_path)}: {str(e)}")
            return False