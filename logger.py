"""Logging functionality for the shopping assistant."""

from collections import deque
import threading
from datetime import datetime
import os

# Store logs in memory
logs = deque(maxlen=100)
log_lock = threading.Lock()

def add_log(message: str, log_type: str = 'info') -> None:
    """
    Add a log entry with timestamp.
    
    Args:
        message: Log message
        log_type: Type of log (info, search, success, error, warning)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'type': log_type
    }
    
    # Print to console with color
    color_code = {
        'info': '\033[36m',     # Cyan
        'search': '\033[34m',   # Blue
        'success': '\033[32m',  # Green
        'error': '\033[31m',    # Red
        'warning': '\033[33m'   # Yellow
    }.get(log_type, '\033[0m')  # Default: no color
    
    print(f"{color_code}[{timestamp}] {message}\033[0m")
    
    # Save to memory
    with log_lock:
        logs.append(log_entry)
        _save_log_to_file(log_entry)

def get_logs(last_timestamp: str = None) -> list:
    """
    Get logs after the specified timestamp.
    
    Args:
        last_timestamp: ISO format timestamp
        
    Returns:
        list: New log entries
    """
    with log_lock:
        if last_timestamp:
            return [log for log in logs if log['timestamp'] > last_timestamp]
        return list(logs)

def clear_logs() -> None:
    """Clear all logs."""
    with log_lock:
        logs.clear()

def _save_log_to_file(log_entry: dict) -> None:
    """Save log entry to file."""
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/shopping_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    
    timestamp = datetime.fromisoformat(log_entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{log_entry['type'].upper()}] {log_entry['message']}\n"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line) 