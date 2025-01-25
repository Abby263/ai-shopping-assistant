# Shopping Assistant

An AI-powered shopping recommendation system that leverages GPT-4o-mini and LangGraph to provide personalized product suggestions through distinct shopper personas. The system combines LLM-based product ideation with real-time web search integration for comprehensive shopping assistance.

## Technical Stack

### Backend Framework
- Flask web framework with session management
- SQLAlchemy ORM for data persistence
- Pydantic for data validation and serialization

### AI/ML Components
- OpenAI GPT-4o-mini for product ideation
- LangGraph for structured AI pipelines
- LangChain for LLM integration and chain management

### Search Integration
- SerpAPI for Google Shopping results
- Custom result scoring and ranking algorithms
- Persona-specific search modifiers

### Authentication
- Flask-Login for session management
- Google OAuth2 integration
- Secure password hashing and validation

### Frontend
- Bootstrap 5 for responsive UI
- AJAX for asynchronous updates
- Custom product card components

## Features

- 🤖 AI-Powered Personas:
  - Trendsetter: Optimizes for novelty and trending items
  - Minimalist: Prioritizes quality and durability scores
  - Savvy: Employs price-performance optimization

- 🔍 Advanced Search Pipeline:
  1. LLM-based product ideation
  2. Structured query generation
  3. Persona-specific result ranking
  4. Dynamic result filtering

- 👤 User Management:
  - OAuth2 and email authentication
  - Session-based user tracking
  - Personalized recommendation engine

- 📊 Data Management:
  - Search history tracking
  - Product view analytics
  - Wishlist functionality
  - Cross-product comparisons

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Google OAuth2 credentials
- SerpAPI API key
- OpenAI API key with GPT-4o-mini access

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd shopping-assistants
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

4. Copy the example environment file and fill in your values:
```bash
cp .env.example .env
```

## Configuration

### Required Environment Variables:
```
# Core Configuration
FLASK_SECRET_KEY=<secure-random-key>
PORT=5001

# Authentication
GOOGLE_CLIENT_ID=<oauth2-client-id>
GOOGLE_CLIENT_SECRET=<oauth2-client-secret>

# API Keys
OPENAI_API_KEY=<gpt4-api-key>
SERPAPI_API_KEY=<serpapi-key>

# Development
FLASK_ENV=development
DEBUG=True
USE_SIMULATE_SEARCH=false
```

## Core Components

### AI Pipeline
```python
def run_shopping_pipeline(persona: str, product_name: str, max_suggestions: int = 3) -> str:
    """
    1. Generate product ideas using GPT-4o-mini
    2. Build optimized search query
    3. Perform web search with persona-specific ranking
    4. Generate HTML results
    """
```

### Shopper Personas
```python
def shopper_enthusiast(state: ShoppingRequest) -> ProductIdeaList:
    """Trendsetter persona optimizing for novelty"""

def shopper_essentialist(state: ShoppingRequest) -> ProductIdeaList:
    """Minimalist persona optimizing for quality"""

def shopper_frugalist(state: ShoppingRequest) -> ProductIdeaList:
    """Savvy persona optimizing for value"""
```

## Project Structure

```
shopping-assistants/
├── app.py              # Main application & AI pipeline
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── .env.example       # Environment template
├── static/            # Static assets
│   └── img/          # Image assets
└── templates/         # Jinja2 templates
    ├── base.html     # Base template with navigation
    ├── index.html    # Search interface
    ├── login.html    # Authentication
    └── signup.html   # User registration
```

## Data Models

The application uses Pydantic models for data validation:
- `ShoppingRequest`: Product search parameters
- `ProductIdea`: AI-generated product suggestions
- `WebSearchResult`: Structured search results
- `UserPreferences`: User data and history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 