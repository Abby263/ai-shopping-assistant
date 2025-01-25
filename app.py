import os
import operator
import re
import json
import logging
import datetime

from flask import Flask, request, render_template, jsonify, session
from typing import Literal, Annotated
from pydantic import BaseModel, Field
from serpapi.google_search import GoogleSearch

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

########################################################
# 1. SETUP
########################################################

app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')

# Enable debug mode for development
app.config['DEBUG'] = True

# Set up session
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-for-testing')
app.config['SESSION_TYPE'] = 'filesystem'

# Ensure templates are auto-reloaded
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Logging
log_formatted_str = "%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_formatted_str)
logger = logging.getLogger(__name__)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Shopper Personas
AI_SHOPPING_PERSONAS = Literal["Trendsetter", "Minimalist", "Savvy"]

########################################################
# 2. DATA MODELS
########################################################

class ShoppingRequest(BaseModel):
    product_name: str
    max_suggestions: int
    min_price: float | None = None
    max_price: float | None = None
    sort_by: Literal["price_low", "price_high", "rating", "popularity"] = "rating"

class ProductIdea(BaseModel):
    name: str = Field(
        description="Name of the Product or Idea."
    )
    description: str = Field(
        description="Description/Reasoning for the Product Suggestion",
    )
    shopper_type: AI_SHOPPING_PERSONAS = Field(
        description="Shopper Type",
    )

class ProductIdeaList(BaseModel):
    ideas: list[ProductIdea]

class SearchQuery(BaseModel):
    shopper_type: AI_SHOPPING_PERSONAS
    search_query: Annotated[str, operator.add]

class WebSearchResult(BaseModel):
    title: str
    link: str
    source: str
    shopper_type: AI_SHOPPING_PERSONAS
    position: int
    thumbnail: str
    price: str
    tag: str
    product_link: str

class WebSearchList(BaseModel):
    search_results: list[WebSearchResult]

class ProductFilters(BaseModel):
    min_price: float | None
    max_price: float | None
    min_rating: float = 3.0
    sort_by: str

class ProductComparison(BaseModel):
    """Model for comparing multiple products."""
    products: list[WebSearchResult]
    comparison_points: dict[str, list[str]]

class UserPreferences(BaseModel):
    """Store user preferences and history."""
    wishlist: list[WebSearchResult] = []
    viewed_products: list[WebSearchResult] = []
    search_history: list[str] = []
    favorite_personas: list[str] = []

########################################################
# 3. SHOPPER PERSONA LOGIC
########################################################

def build_product_ideas(state: ShoppingRequest, shopper_type: AI_SHOPPING_PERSONAS, instructions: str) -> ProductIdeaList:
    """
    Uses the LLM to generate product ideas for the requested product name,
    given the user-chosen shopper persona instructions.
    """
    logger.info(f"Generating product ideas for persona={shopper_type}")
    system_message = instructions.format(
        product_name=state.product_name, 
        max_suggestions=state.max_suggestions
    )
    structured_llm = llm.with_structured_output(ProductIdeaList)

    # Prompt the LLM
    llm_response = structured_llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Generate {state.max_suggestions} product suggestions.")
        ]
    )
    return llm_response

def shopper_enthusiast(state: ShoppingRequest) -> ProductIdeaList:
    """
    Persona that prioritizes excitement, novelty, or fun in products.
    """
    instructions = """You are a trendsetting shopper with an eye for the latest and greatest,
seeking products that spark joy and create unforgettable experiences. You love discovering
cutting-edge items that make life more vibrant and exciting.

Generate {max_suggestions} product suggestions for "{product_name}".
"""
    return build_product_ideas(state, "Trendsetter", instructions)

def shopper_essentialist(state: ShoppingRequest) -> ProductIdeaList:
    """
    Persona that focuses on functional, purposeful items.
    """
    instructions = """You are a minimalist shopper who values quality and purpose,
choosing items that bring lasting value and elegant simplicity to life.
Each recommendation focuses on timeless design and essential functionality.

Generate {max_suggestions} product suggestions for "{product_name}".
"""
    return build_product_ideas(state, "Minimalist", instructions)

def shopper_frugalist(state: ShoppingRequest) -> ProductIdeaList:
    """
    Persona that emphasizes cost-effective or budget-friendly options.
    """
    instructions = """You are a savvy shopper with a talent for finding hidden gems
and incredible deals. You excel at discovering high-quality products that offer
exceptional value without compromising on quality.

Generate {max_suggestions} product suggestions for "{product_name}".
"""
    return build_product_ideas(state, "Savvy", instructions)

########################################################
# 4. SEARCH AND RESULT GENERATION
########################################################

def scour_the_internet(idea_list: ProductIdeaList) -> SearchQuery:
    """
    Prepares a detailed and targeted search query from the product ideas.
    Combines key aspects of the product with specific criteria based on shopper type.
    """
    if not idea_list.ideas:
        return SearchQuery(shopper_type="Enthusiast", search_query="(No ideas)")
    
    first_idea = idea_list.ideas[0]
    
    # Add persona-specific search modifiers
    modifiers = {
        "Trendsetter": "new trending latest",
        "Minimalist": "essential high-quality durable",
        "Savvy": "best value price-performance"
    }
    
    # Extract key features from description
    key_features = re.findall(r'[\w\s-]+(?=\band\b|\bor\b|,|\.|$)', first_idea.description)
    features_str = ' '.join(key_features[:2]) if key_features else first_idea.description
    
    # Build enhanced query
    query_str = f"{first_idea.name} {features_str} {modifiers.get(first_idea.shopper_type, '')}"
    query_str = re.sub(r'\s+', ' ', query_str).strip()  # Clean up whitespace
    
    return SearchQuery(shopper_type=first_idea.shopper_type, search_query=query_str)

def web_search_agent(state: SearchQuery) -> WebSearchList:
    """
    Uses SerpAPI or simulated results to fetch product deals with enhanced filtering and ranking.
    """
    logger.info(f"Search query for persona={state.shopper_type}: {state.search_query}")
    
    all_results = []
    use_simulate = os.getenv("USE_SIMULATE_SEARCH", "false").lower() == "true"
    
    if use_simulate:
        # Return some dummy results
        all_results = [
            WebSearchResult(
                title="Sample Deal 1",
                link="http://example.com/deal1",
                source="Example",
                shopper_type=state.shopper_type,
                position=1,
                thumbnail="http://example.com/thumbnail1.jpg",
                price="$9.99",
                tag="Budget",
                product_link="http://example.com/product1"
            ),
            WebSearchResult(
                title="Sample Deal 2",
                link="http://example.com/deal2",
                source="Example",
                shopper_type=state.shopper_type,
                position=2,
                thumbnail="http://example.com/thumbnail2.jpg",
                price="$25.00",
                tag="Popular",
                product_link="http://example.com/product2"
            ),
            WebSearchResult(
                title="Sample Deal 3",
                link="http://example.com/deal3",
                source="Example",
                shopper_type=state.shopper_type,
                position=3,
                thumbnail="http://example.com/thumbnail3.jpg",
                price="$30.00",
                tag="Favorite",
                product_link="http://example.com/product3"
            ),
            WebSearchResult(
                title="Sample Deal 4",
                link="http://example.com/deal4",
                source="Example",
                shopper_type=state.shopper_type,
                position=4,
                thumbnail="http://example.com/thumbnail4.jpg",
                price="$40.00",
                tag="Trending",
                product_link="http://example.com/product4"
            ),
            WebSearchResult(
                title="Sample Deal 5",
                link="http://example.com/deal5",
                source="Example",
                shopper_type=state.shopper_type,
                position=5,
                thumbnail="http://example.com/thumbnail5.jpg",
                price="$50.00",
                tag="Best Value",
                product_link="http://example.com/product5"
            ),
            WebSearchResult(
                title="Sample Deal 6",
                link="http://example.com/deal6",
                source="Example",
                shopper_type=state.shopper_type,
                position=6,
                thumbnail="http://example.com/thumbnail6.jpg",
                price="$60.00",
                tag="Premium",
                product_link="http://example.com/product6"
            ),
        ]
    else:
        # Real SerpAPI calls with enhanced parameters
        serpapi_api_key = os.getenv("SERPAPI_API_KEY", "")
        params = {
            "q": state.search_query,
            "api_key": serpapi_api_key,
            "engine": "google_shopping",
            "google_domain": "google.com",
            "direct_link": "true",
            "gl": "us",
            "hl": "en",
            "num": "20",  # Fetch more results for better filtering
            "sort": "review_score" if state.shopper_type == "Minimalist" else "price_low_to_high" if state.shopper_type == "Savvy" else "review_count"
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        # Enhanced result processing with filtering and scoring
        processed_results = []
        for idx, item in enumerate(results.get("shopping_results", []), start=1):
            # Skip results without essential information
            if not all([item.get("title"), item.get("link"), item.get("price")]):
                continue

            # Calculate result score based on persona
            score = 0
            if state.shopper_type == "Trendsetter":
                score += float(item.get("rating", 0)) * 2  # Weight ratings more heavily
                score += float(item.get("review_count", 0)) * 0.01  # Consider popularity
            elif state.shopper_type == "Minimalist":
                score += float(item.get("rating", 0)) * 3  # Quality is paramount
                if "warranty" in item.get("description", "").lower():
                    score += 2  # Bonus for items with warranty
            elif state.shopper_type == "Savvy":
                price = parse_price(item.get("price", "inf"))
                if price == float("inf"):
                    continue
                score += (1000 / price) if price > 0 else 0  # Higher score for lower prices
                score += float(item.get("rating", 0))  # Still consider quality

            processed_results.append((
                WebSearchResult(
                    title=item.get("title", "No Title"),
                    link=item.get("link", "No Link"),
                    source=item.get("source", "Unknown"),
                    shopper_type=state.shopper_type,
                    position=idx,
                    thumbnail=item.get("thumbnail", ""),
                    price=item.get("price", "No Price"),
                    tag=get_result_tag(item, state.shopper_type),
                    product_link=item.get("product_link", "")
                ),
                score
            ))

        # Sort by score and take top results
        processed_results.sort(key=lambda x: x[1], reverse=True)
        all_results = [result for result, _ in processed_results]

    return WebSearchList(search_results=all_results)

def get_result_tag(item: dict, shopper_type: str) -> str:
    """
    Generate meaningful tags based on product attributes and shopper type.
    """
    rating = float(item.get("rating", 0))
    review_count = int(item.get("review_count", 0))
    price = parse_price(item.get("price", "inf"))
    
    if shopper_type == "Trendsetter":
        if review_count > 1000 and rating >= 4.5:
            return "Top Rated"
        elif "new" in item.get("title", "").lower():
            return "New Arrival"
        return "Trending"
    elif shopper_type == "Minimalist":
        if rating >= 4.7:
            return "Premium Quality"
        elif "warranty" in item.get("description", "").lower():
            return "Guaranteed"
        return "Essential"
    else:  # Savvy
        if price < 20:
            return "Great Deal"
        elif rating >= 4.5 and review_count > 500:
            return "Best Value"
        return "Smart Choice"

def parse_price(price_str: str) -> float:
    """
    Attempts to parse a price string like "$25.99" -> 25.99
    """
    match = re.search(r"\$?([\d,]+(\.\d+)?)", price_str)
    if match:
        return float(match.group(1).replace(",", ""))
    return float("inf")

########################################################
# 5. BUILDING THE BOOTSTRAP HTML
########################################################

def build_html_results(results: WebSearchList, persona: str, max_suggestions: int) -> str:
    """
    Builds a Bootstrap-based HTML card layout from the search results.
    """
    persona_results = [r for r in results.search_results 
                      if r.shopper_type.lower() == persona.lower()]
    persona_results.sort(key=lambda x: parse_price(x.price))
    top_results = persona_results[:max_suggestions]

    if not top_results:
        return '<div class="alert alert-info">No results found.</div>'

    html_output = f"""
    <div class="card">
        <div class="card-header">
            <h3 class="mb-0">Top Suggestions for {persona}</h3>
        </div>
        <div class="card-body">
            <div class="row">
    """

    for result in top_results:
        html_output += f"""
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card product-card">
                    <span class="product-tag">{result.tag}</span>
                    <img src="{result.thumbnail}" class="product-image" alt="{result.title}">
                    <div class="card-body">
                        <h5 class="card-title">{result.title}</h5>
                        <p class="product-price">{result.price}</p>
                        <p class="card-text text-muted">From {result.source}</p>
                        <a href="{result.link}" target="_blank" class="btn btn-primary w-100">View Deal</a>
                    </div>
                </div>
            </div>
        """

    html_output += """
            </div>
        </div>
    </div>
    """
    return html_output

########################################################
# 6. END-TO-END PIPELINE
########################################################

def run_shopping_pipeline(persona: str, product_name: str, max_suggestions: int = 3) -> str:
    """
    1) Generate product ideas for the chosen persona.
    2) Build query from the first idea.
    3) Perform web search.
    4) Build & return HTML results.
    """
    req = ShoppingRequest(product_name=product_name, max_suggestions=max_suggestions)

    # Map persona string to the correct function
    persona_funcs = {
        "Trendsetter": shopper_enthusiast,
        "Minimalist": shopper_essentialist,
        "Savvy": shopper_frugalist
    }
    if persona not in persona_funcs:
        raise ValueError("Invalid persona selected.")

    # 1) Generate product ideas
    idea_list = persona_funcs[persona](req)

    # 2) Build search query
    search_query = scour_the_internet(idea_list)

    # 3) Perform the web search
    results = web_search_agent(search_query)

    # 4) Build the HTML for results
    html_results = build_html_results(results, persona, max_suggestions)
    return html_results

########################################################
# 7. FLASK ROUTES
########################################################

@app.route("/", methods=["GET"])
def index():
    """
    Render the main page with the search form.
    """
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    """
    Handle form submission, run the pipeline, and return results.
    """
    persona = request.form.get("shopper_type", "Trendsetter")
    product_name = request.form.get("product_name", "")
    max_suggestions = int(request.form.get("max_suggestions", 3))

    # Run the shopping pipeline
    results_html = run_shopping_pipeline(persona, product_name, max_suggestions)
    
    # Store results in session for comparison and tracking
    if hasattr(results_html, 'search_results'):
        session['last_results'] = [result.dict() for result in results_html.search_results]
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"html": results_html})
    
    return render_template("index.html", results_html=results_html)

@app.route("/compare", methods=["POST"])
def compare_products():
    """
    Handle product comparison requests.
    """
    product_ids = request.form.getlist("product_ids[]")
    products = [p for p in session.get('last_results', []) if str(p.position) in product_ids]
    
    if len(products) < 2:
        return jsonify({"error": "Please select at least 2 products to compare"})
    
    comparison = generate_comparison(products)
    return render_template("comparison.html", comparison=comparison)

def generate_comparison(products: list[WebSearchResult]) -> ProductComparison:
    """
    Generate a detailed comparison between selected products.
    """
    comparison_points = {
        "Price Range": [p.price for p in products],
        "Source": [p.source for p in products],
        "Rating": [str(getattr(p, 'rating', 'N/A')) for p in products],
        "Reviews": [str(getattr(p, 'review_count', 'N/A')) for p in products],
        "Tags": [p.tag for p in products],
    }
    
    return ProductComparison(
        products=products,
        comparison_points=comparison_points
    )

def build_comparison_html(comparison: ProductComparison) -> str:
    """
    Build HTML for product comparison view.
    """
    html = """
    <div class="comparison-table">
        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>Feature</th>
    """
    
    # Add product names to header
    for product in comparison.products:
        html += f"<th>{product.title}</th>"
    html += "</tr></thead><tbody>"
    
    # Add comparison points
    for feature, values in comparison.comparison_points.items():
        html += f"<tr><td><strong>{feature}</strong></td>"
        for value in values:
            html += f"<td>{value}</td>"
        html += "</tr>"
    
    html += "</tbody></table></div>"
    return html

def init_user_session():
    """Initialize user session with preferences."""
    if 'user_prefs' not in session:
        session['user_prefs'] = UserPreferences().dict()

@app.route("/wishlist/add", methods=["POST"])
def add_to_wishlist():
    """Add a product to the wishlist."""
    init_user_session()
    product_id = request.form.get("product_id")
    products = session.get('last_results', [])
    product = next((p for p in products if str(p.position) == product_id), None)
    
    if product:
        prefs = UserPreferences(**session['user_prefs'])
        prefs.wishlist.append(product)
        session['user_prefs'] = prefs.dict()
        return jsonify({"success": True})
    return jsonify({"error": "Product not found"})

@app.route("/history", methods=["GET"])
def view_history():
    """View search and product history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    return render_template(
        "history.html",
        viewed_products=prefs.viewed_products[-10:],  # Last 10 viewed
        search_history=prefs.search_history[-10:],    # Last 10 searches
        wishlist=prefs.wishlist
    )

def track_product_view(product: WebSearchResult):
    """Track when a product is viewed."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    prefs.viewed_products.append(product)
    if len(prefs.viewed_products) > 50:  # Keep last 50 views
        prefs.viewed_products = prefs.viewed_products[-50:]
    session['user_prefs'] = prefs.dict()

def track_search(query: str):
    """Track search history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    prefs.search_history.append(query)
    if len(prefs.search_history) > 50:  # Keep last 50 searches
        prefs.search_history = prefs.search_history[-50:]
    session['user_prefs'] = prefs.dict()

@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    """Get personalized recommendations based on history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    
    # Analyze viewing history and wishlist to generate recommendations
    viewed_categories = set()
    price_ranges = []
    
    for product in prefs.viewed_products + prefs.wishlist:
        viewed_categories.add(product.tag)
        price_ranges.append(parse_price(product.price))
    
    # Calculate preferred price range
    avg_price = sum(price_ranges) / len(price_ranges) if price_ranges else 0
    
    # Generate recommendations based on preferences
    recommendations = []
    for category in viewed_categories:
        req = ShoppingRequest(
            product_name=f"products in {category}",
            max_suggestions=2,
            min_price=avg_price * 0.7,
            max_price=avg_price * 1.3
        )
        results = run_shopping_pipeline(prefs.favorite_personas[0] if prefs.favorite_personas else "Trendsetter", req.product_name)
        recommendations.extend(results)
    
    return render_template("recommendations.html", recommendations=recommendations)

@app.route("/track/search", methods=["POST"])
def track_search():
    """Track user search queries."""
    query = request.form.get("query", "")
    if query:
        track_search_history(query)
    return jsonify({"success": True})

@app.route("/track/product", methods=["POST"])
def track_product():
    """Track product views."""
    product_id = request.form.get("product_id")
    products = session.get('last_results', [])
    product = next((p for p in products if str(p.position) == product_id), None)
    
    if product:
        track_product_view(product)
        return jsonify({"success": True})
    return jsonify({"error": "Product not found"})

def track_search_history(query: str):
    """Track search history."""
    init_user_session()
    prefs = UserPreferences(**session['user_prefs'])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    prefs.search_history.append(f"{query} ({timestamp})")
    if len(prefs.search_history) > 50:  # Keep last 50 searches
        prefs.search_history = prefs.search_history[-50:]
    session['user_prefs'] = prefs.dict()

########################################################
# 8. MAIN
########################################################

if __name__ == "__main__":
    # By default runs on port 5000
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
