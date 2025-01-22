"""
AI Shopping Assistant with advanced research capabilities
"""

from typing import List, Dict, Optional, Annotated, Sequence, TypedDict
from datetime import datetime
import os
import json
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from logger import add_log

# Load environment variables
load_dotenv()

# Initialize search tool with specific parameters for product search
tavily_search = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ProductResearchResult(TypedDict):
    """Structure for product research results."""
    title: str
    price: float
    url: str
    description: str
    image_url: str
    source: str
    analysis: Dict

# Create product research agent
product_research_agent = create_react_agent(
    llm,
    tools=[tavily_search],
    state_modifier="""You are an expert shopping research assistant. Your task is to find and analyze products matching the user's criteria.

    IMPORTANT INSTRUCTIONS:
    1. Search for specific products with accurate prices
    2. Analyze each product for:
       - Price competitiveness
       - Product features
       - Retailer reliability
       - Deal quality
    3. Format data for visualization:
       - Price comparison (bar chart)
       - Features comparison (table)
       - Retailer distribution (pie chart)
    4. Return data in this exact format:
    {
        "products": [
            {
                "title": "Product Name",
                "price": float_price,
                "url": "product_url",
                "features": ["feature1", "feature2"],
                "retailer": "store_name",
                "deal_score": float_0_to_10
            }
        ],
        "analysis": {
            "price_range": {"min": float, "max": float, "avg": float},
            "best_deals": ["product1", "product2"],
            "recommended_charts": ["bar_chart_prices", "pie_chart_retailers"]
        }
    }

    Focus on finding specific product pages, not category listings.
    """
)

@tool
def search_products(query: str) -> dict:
    """Search for products matching user query."""
    try:
        add_log(f"Searching for products: {query}", 'info')
        
        # Create a more focused search query
        search_query = f"{query} product price buy"
        add_log(f"Using search query: {search_query}", 'info')
        
        # Get search results
        results = tavily_search.invoke(search_query)
        add_log(f"Received {len(results)} raw results", 'info')
        
        # Process results
        products = []
        for result in results:
            # Log raw result for debugging
            add_log(f"Processing result: {result.get('title', 'No title')} | URL: {result.get('url', 'No URL')}", 'debug')
            
            # Extract price from content
            content = result.get('content', '')
            price = _extract_price(content)
            
            if price:
                product = {
                    "title": result.get('title', '').split(' - ')[0].strip(),
                    "url": result.get('url', ''),
                    "price": price,
                    "description": _extract_description(content),
                    "image_url": result.get('image_url', ''),
                    "source": result.get('source', '').replace('.com', '')
                }
                products.append(product)
                add_log(f"Found valid product: {product['title']} at ${price}", 'success')
        
        # Sort by price and get top 5
        sorted_products = sorted(products, key=lambda x: x['price'])[:5]
        
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "products": sorted_products,
            "total_results": len(sorted_products)
        }
        
        add_log(f"Found {len(sorted_products)} products", 'success')
        return results
        
    except Exception as e:
        error_msg = f"Error in search: {str(e)}"
        add_log(error_msg, 'error')
        return _format_error(query, error_msg)



def _extract_price(content: str) -> Optional[float]:
    """Extract price from content using multiple patterns."""
    if not content:
        return None
        
    # Price patterns from most to least specific
    price_patterns = [
        r'\$(\d+(?:\.\d{2})?)',  # Standard price format
        r'Price:\s*\$?(\d+(?:\.\d{2})?)',  # Labeled price
        r'(\d+(?:\.\d{2})?)\s*dollars',  # Text price
        r'USD\s*(\d+(?:\.\d{2})?)',  # USD format
        r'(?<!\$)(\d+(?:\.\d{2})?)\s*$'  # Number at end of string
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, content)
        if matches:
            # Convert all matches to floats and filter valid prices
            prices = []
            for match in matches:
                try:
                    price = float(match)
                    if 0.01 <= price <= 10000:  # Reasonable price range
                        prices.append(price)
                except ValueError:
                    continue
            
            if prices:
                return min(prices)  # Return lowest valid price
    
    return None

def _extract_description(content: str) -> str:
    """Extract product description from content with basic sanitization."""
    if not content:
        return ""
    
    # Clean and trim the description to keep the most relevant details
    description = re.sub(r'\s+', ' ', content.strip())
    return description[:500]  # Limit description to 500 characters

def _format_error(query: str, error_msg: str) -> dict:
    """Format error response."""
    return {
        "error": error_msg,
        "query": query,
        "timestamp": datetime.now().isoformat()
    }

def run_shopping_assistant(query: str) -> dict:
    """Run the shopping assistant."""
    try:
        add_log(f"Starting search for: {query}", 'info')
        return search_products(query)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        add_log(error_msg, 'error')
        return _format_error(query, error_msg)
