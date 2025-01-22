# AI Shopping Assistant

An intelligent shopping assistant powered by LangGraph-based AI agents that helps users find the best deals across multiple retailers. The system uses advanced natural language processing through OpenAI's GPT models and Tavily's search capabilities to understand user queries, perform comprehensive product research, and provide detailed recommendations with price comparisons.

## Architecture

The application uses a multi-agent system built with LangGraph to orchestrate different specialized AI agents:

### Core Agents

1. **Query Understanding Agent**
   - Powered by OpenAI GPT-4
   - Parses natural language queries into structured search parameters
   - Extracts key product attributes, price ranges, and preferences
   - Handles disambiguation when needed

2. **Search Agent**
   - Leverages Tavily Search API for real-time product discovery
   - Performs targeted searches across multiple e-commerce platforms
   - Filters and validates search results
   - Handles pagination and result aggregation

3. **Analysis Agent**
   - Compares products based on multiple criteria
   - Performs price analysis and trend detection
   - Identifies best deals and value propositions
   - Generates product summaries and recommendations

4. **Response Agent**
   - Formats and structures the final response
   - Prioritizes relevant information
   - Generates natural language explanations
   - Handles error cases and fallbacks

### Agent Workflow

1. User query → Query Understanding Agent
2. Structured parameters → Search Agent
3. Raw results → Analysis Agent
4. Analyzed data → Response Agent
5. Final response → User

## Features

- Natural language query processing for intuitive product search
- Multi-retailer product search and aggregation
- Smart price comparison and trend analysis
- Personalized best deal recommendations based on user preferences
- Detailed product information extraction including specifications and reviews
- Real-time search results with minimal latency
- Logging system for tracking and debugging
- Clean and responsive web interface
- RESTful API for integration with other services

## Project Structure

```
.
├── app.py              # Main Flask application
├── agent.py            # AI agent implementation and product search logic
├── logger.py           # Logging configuration and utilities
├── requirements.txt    # Python dependencies
├── .env               # Environment variables configuration
├── static/            # Static assets (CSS, JS, images)
├── templates/         # HTML templates
├── logs/              # Application logs
└── outputs/           # Generated outputs and cached results
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Tavily API key
- Internet connection for real-time product searches

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-shopping-assistant.git
cd ai-shopping-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with the following configuration:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
LOG_LEVEL=INFO  # Optional: Set to DEBUG for more detailed logs
```

## Configuration

### API Configuration
The application requires API keys for the following services:
- **OpenAI API**: Powers the natural language processing and agent intelligence
  - Required scopes: text-generation, embeddings
  - Rate limits vary by subscription tier
- **Tavily API**: Enables intelligent product search
  - Default rate limit: 60 requests per minute
  - Supports concurrent searches

### Logging
The application uses a structured logging system with the following features:
- Log rotation with daily files
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Detailed error tracking with timestamps
- Log files are stored in the `logs/` directory

## Usage

1. Start the server:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:5000`

3. Enter your shopping query in natural language, for example:
- "Find me the best deal on Blue Nike running shoes under $100"
- "Compare prices for Red Puma t-shirts within $30"
- "Show me gaming laptops under $1500 with RTX 4060"
- "Find deals on wireless earbuds with noise cancellation"

The assistant will:
1. Analyze your query for key product attributes
2. Search across multiple retailers
3. Compare prices and features
4. Present the best deals with detailed information

## API Reference

### POST /search
Search for products based on user query.

**Request Body:**
```json
{
    "query": "string"
}
```

**Response:**
```json
{
    "query": "string",
    "timestamp": "string",
    "products": [
        {
            "title": "string",
            "price": number,
            "url": "string",
            "description": "string",
            "image_url": "string",
            "source": "string"
        }
    ],
    "best_deal": {
        "title": "string",
        "price": number,
        "url": "string",
        "description": "string",
        "image_url": "string",
        "source": "string"
    },
    "total_results": number,
    "error": null | string
}
```

### Error Codes
- 200: Successful search
- 400: Invalid request parameters
- 401: Authentication error
- 429: Rate limit exceeded
- 500: Internal server error

## Technologies Used

- **AI/ML Framework**:
  - LangGraph for agent orchestration and workflow management
  - LangChain for AI component integration
  - OpenAI GPT-4 for natural language understanding and generation
  - Tavily Search API for intelligent web search and data extraction
- **Backend Framework**: Flask (Python web framework)
- **Frontend**:
  - Bootstrap 5 for responsive design
  - JavaScript for dynamic interactions
- **Development Tools**:
  - Python virtual environment
  - Git for version control
  - Environment variables for configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

Common issues and solutions:
1. **API Key Issues**: Ensure your API keys are correctly set in the `.env` file
2. **Rate Limits**: Check the logs for rate limit errors and adjust request timing
3. **Search Errors**: Verify internet connectivity and API service status

## License

MIT License - See LICENSE file for details

## Support

For issues and feature requests, please create an issue in the repository.