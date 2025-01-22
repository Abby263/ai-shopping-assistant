from flask import Flask, render_template, request, jsonify
from agent import run_shopping_assistant
from logger import add_log, get_logs
import json
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        add_log(f"Received search query: {query}", 'info')
        
        # Run shopping assistant
        results = run_shopping_assistant(query)
        
        if results.get("error"):
            add_log(f"Error: {results['error']}", 'error')
            return jsonify({
                'error': results["error"],
                'query': query
            }), 404

        add_log(f"Found {results.get('total_results', 0)} products", 'success')
        return jsonify(results)

    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        add_log(error_msg, 'error')
        return jsonify({'error': error_msg}), 500

@app.route('/logs', methods=['POST'])
def get_new_logs():
    """Return new logs since the last timestamp."""
    try:
        data = request.json
        last_timestamp = data.get('last_timestamp')
        new_logs = get_logs(last_timestamp)
        return jsonify({'logs': new_logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 