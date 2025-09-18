"""
Logger utility functions for eConsult Analysis Platform.
Provides helper functions for consistent logging across the application.
"""

import time
import logging
import os
import sys
import platform
import functools

# Get the main logger that's already configured in main.py
logger = logging.getLogger("eConsult")

def log_function_call(func):
    """
    Decorator to log function calls, arguments, and execution time.
    
    Usage:
    @log_function_call
    def my_function(arg1, arg2):
        # function code here
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Format arguments for logging, but limit their size
        args_repr = [repr(a)[:100] + ('...' if len(repr(a)) > 100 else '') for a in args]
        kwargs_repr = [f"{k}={repr(v)[:100] + ('...' if len(repr(v)) > 100 else '')}" 
                      for k, v in kwargs.items()]
        
        # Combine positional and keyword arguments
        all_args = ", ".join(args_repr + kwargs_repr)
        
        # Log the function call
        logger.info(f"CALLING: {func.__name__}({all_args})")
        
        # Time the function execution
        start_time = time.time()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Log successful completion and execution time
            execution_time = time.time() - start_time
            logger.info(f"COMPLETED: {func.__name__} in {execution_time:.4f}s")
            
            return result
            
        except Exception as e:
            # Log exception and execution time
            execution_time = time.time() - start_time
            logger.error(f"ERROR in {func.__name__} after {execution_time:.4f}s: {str(e)}", 
                       exc_info=True)
            raise
            
    return wrapper

def log_api_call(api_name):
    """
    Decorator to log API calls with detailed timing and response info.
    
    Usage:
    @log_api_call("Ollama")
    def call_ollama_api(prompt):
        # API call code here
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log the API call
            logger.info(f"API CALL: {api_name} via {func.__name__}")
            
            # Time the API call
            start_time = time.time()
            
            try:
                # Call the original function
                result = func(*args, **kwargs)
                
                # Log successful completion and execution time
                execution_time = time.time() - start_time
                logger.info(f"API RESPONSE: {api_name} responded in {execution_time:.4f}s")
                
                # Log response size if result is a dict or string
                if isinstance(result, dict):
                    logger.info(f"Response size: {sys.getsizeof(result)} bytes")
                elif isinstance(result, str):
                    logger.info(f"Response length: {len(result)} characters")
                
                return result
                
            except Exception as e:
                # Log exception and execution time
                execution_time = time.time() - start_time
                logger.error(f"API ERROR: {api_name} failed after {execution_time:.4f}s: {str(e)}", 
                           exc_info=True)
                raise
                
        return wrapper
    return decorator

def log_process_start(process_name, additional_info=None):
    """
    Log the start of a major process with visual separation and system info.
    
    Args:
        process_name: Name of the process that's starting
        additional_info: Optional dictionary of additional information to log
    """
    logger.info("="*50)
    logger.info(f"STARTING PROCESS: {process_name.upper()}")
    logger.info("="*50)
    
    # Log timestamp
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log additional info if provided
    if additional_info:
        for key, value in additional_info.items():
            logger.info(f"{key}: {value}")
    
    # Return start time for duration calculation
    return time.time()

def log_process_end(process_name, start_time, metrics=None):
    """
    Log the end of a major process with timing information.
    
    Args:
        process_name: Name of the process that's ending
        start_time: Start time returned from log_process_start
        metrics: Optional dictionary of metrics to log
    """
    # Calculate duration
    duration = time.time() - start_time
    
    logger.info("-"*50)
    logger.info(f"COMPLETED PROCESS: {process_name.upper()}")
    logger.info(f"Duration: {duration:.2f} seconds")
    
    # Log metrics if provided
    if metrics:
        logger.info("Process metrics:")
        for key, value in metrics.items():
            logger.info(f"  - {key}: {value}")
    
    logger.info("="*50)

def log_error(error_message, exception=None, context=None):
    """
    Standardized error logging with optional context information.
    
    Args:
        error_message: Main error message
        exception: Exception object if available
        context: Dictionary of contextual information
    """
    logger.error("!"*50)
    logger.error(f"ERROR: {error_message}")
    
    if context:
        logger.error("Error context:")
        for key, value in context.items():
            logger.error(f"  - {key}: {value}")
    
    if exception:
        logger.error(f"Exception: {type(exception).__name__}: {str(exception)}")
        logger.error("Stack trace:", exc_info=True)
    
    logger.error("!"*50)

def get_system_info():
    """
    Get system information for diagnostic logging.
    Returns a dictionary of system information.
    """
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'os': f"{platform.system()} {platform.release()}",
        'machine': platform.machine(),
        'processor': platform.processor(),
        'hostname': platform.node()
    }
    
    # Try to add memory info if possible
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory_total'] = f"{mem.total / (1024**3):.2f} GB"
        info['memory_available'] = f"{mem.available / (1024**3):.2f} GB"
        info['memory_percent'] = f"{mem.percent}%"
        
        # CPU info
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_percent'] = f"{psutil.cpu_percent()}%"
    except ImportError:
        # psutil not available, just continue without memory info
        pass
    
    return info

def log_file_operation(operation, file_path, status="success", error=None, size=None):
    """
    Log file operations (read, write, delete, etc.) with consistent formatting.
    
    Args:
        operation: Type of operation (read, write, delete, etc.)
        file_path: Path to the file
        status: Operation status (success, failed)
        error: Error message if operation failed
        size: File size in bytes if applicable
    """
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    
    if status == "success":
        log_level = logging.INFO
        message = f"FILE {operation.upper()}: {file_name}"
    else:
        log_level = logging.ERROR
        message = f"FILE {operation.upper()} FAILED: {file_name}"
    
    # Build additional info
    info = [
        f"directory: {file_dir}",
    ]
    
    if size is not None:
        # Format size based on magnitude
        if size < 1024:
            size_str = f"{size} bytes"
        elif size < 1024**2:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size/(1024**2):.2f} MB"
        info.append(f"size: {size_str}")
    
    if error:
        info.append(f"error: {error}")
    
    # Log the message
    logger.log(log_level, f"{message} ({', '.join(info)})")