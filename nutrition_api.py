import requests
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env variables into environment

api_key = os.getenv('USDA_API_KEY')


class USDANutritionAPI:
    """
    Client for USDA FoodData Central API.
    
    API Documentation: https://fdc.nal.usda.gov/api-guide.html
    """
    
    BASE_URL = "https://api.nal.usda.gov/fdc/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            api_key: USDA API key (get from https://fdc.nal.usda.gov/api-key-signup.html)
                     If None, looks for USDA_API_KEY environment variable
        
        Why authentication?
        - Prevents abuse
        - Tracks usage for rate limiting
        - Free tier: 1000 requests/hour
        """
        self.api_key = api_key or os.environ.get('USDA_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API key required. Get one at https://fdc.nal.usda.gov/api-key-signup.html\n"
                "Pass as argument or set USDA_API_KEY environment variable"
            )
        
        self.session = requests.Session()  # Reuse connections for efficiency
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests (rate limiting)
    
    def _rate_limit(self):
        """
        Enforce minimum time between requests.
        
        Why rate limiting?
        - Prevents exceeding API quota
        - Avoids 429 "Too Many Requests" errors
        - Be a good API citizen
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def search_food(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for foods matching the query.
        
        Args:
            query: Food name (e.g., "apple", "chicken breast")
            max_results: Maximum number of results to return
        
        Returns:
            List of dictionaries with keys: fdc_id, description, data_type
        
        Data Types:
        - "Foundation": Basic foods (apple, milk, etc.)
        - "SR Legacy": USDA Standard Reference
        - "Branded": Commercial products with barcodes
        - "Survey (FNDDS)": Survey foods (recipes, mixed dishes)
        """
        self._rate_limit()
        
        endpoint = f"{self.BASE_URL}/foods/search"
        
        params = {
            'api_key': self.api_key,
            'query': query,
            'pageSize': max_results,
            'dataType': ['Foundation', 'SR Legacy'],  # Prioritize basic foods
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()  # Raise exception for 4xx/5xx status codes
            
            data = response.json()
            foods = data.get('foods', [])
            
            return [{
                'fdc_id': food['fdcId'],
                'description': food['description'],
                'data_type': food.get('dataType', 'Unknown'),
                'score': food.get('score', 0)  # Relevance score
            } for food in foods]
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return []
    
    def get_food_details(self, fdc_id: int) -> Optional[Dict]:
        """
        Retrieve detailed nutritional information for a specific food.
        
        Args:
            fdc_id: FoodData Central ID (from search results)
        
        Returns:
            Dictionary with nutritional data or None if request fails
        
        Response includes:
        - description: Food name
        - foodNutrients: List of nutrient objects
        - servingSize: Portion size
        - householdServingFullText: Common serving (e.g., "1 cup")
        """
        self._rate_limit()
        
        endpoint = f"{self.BASE_URL}/food/{fdc_id}"
        
        params = {
            'api_key': self.api_key,
            'format': 'full'  # Include all nutrient data
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get food details: {e}")
            return None
    
    def parse_nutrients(self, food_details: Dict) -> Dict:
        """
        Extract key nutrients from API response.
        
        Why these nutrients?
        - Energy (calories): Primary metric for calorie counting
        - Macronutrients (protein, fat, carbs): Essential for diet planning
        - Fiber: Important for digestive health
        - Sugars: Useful for diabetics and low-sugar diets
        
        All values per 100g for standardization.
        """
        if not food_details:
            return {}
        
        # Map nutrient names to keys
        nutrient_map = {
            'Energy': 'calories',
            'Protein': 'protein',
            'Total lipid (fat)': 'fat',
            'Carbohydrate, by difference': 'carbohydrates',
            'Fiber, total dietary': 'fiber',
            'Sugars, total including NLEA': 'sugars',
            'Sodium, Na': 'sodium',
            'Cholesterol': 'cholesterol',
        }
        
        nutrients = {}
        
        # Extract nutrients from foodNutrients array
        for nutrient in food_details.get('foodNutrients', []):
            name = nutrient.get('nutrient', {}).get('name')
            
            if name in nutrient_map:
                key = nutrient_map[name]
                amount = nutrient.get('amount', 0)
                unit = nutrient.get('nutrient', {}).get('unitName', '')
                
                nutrients[key] = {
                    'amount': round(amount, 2),
                    'unit': unit
                }
        
        # Add metadata
        nutrients['food_name'] = food_details.get('description', 'Unknown')
        nutrients['serving_size'] = food_details.get('servingSize', 100)
        
        # Sometimes serving size unit is under 'servingSizeUnit'; fallback to empty string if missing
        nutrients['serving_unit'] = food_details.get('servingSizeUnit', '')
        
        # Optional: household serving description
        nutrients['household_serving'] = food_details.get('householdServingFullText', '')
        
        return nutrients
    
    def get_nutrition_for_food(self, food_name: str) -> Optional[Dict]:
        """
        One-step function: search food and return nutrition.
        
        This is the primary method most users will call.
        It combines search + retrieve + parse into single operation.
        """
        # Search for the food
        search_results = self.search_food(food_name, max_results=1)
        
        if not search_results:
            print(f"❌ No results found for '{food_name}'")
            return None
        
        # Get the top result
        top_result = search_results[0]
        print(f"✅ Found: {top_result['description']} (score: {top_result['score']:.2f})")
        
        # Retrieve detailed nutrition
        food_details = self.get_food_details(top_result['fdc_id'])
        
        if not food_details:
            return None
        
        # Parse and return nutrients
        return self.parse_nutrients(food_details)


# Utility function for name mapping
def map_imagenet_to_usda(imagenet_name: str) -> str:
    """
    Map ImageNet class names to USDA-friendly search terms.
    
    Why needed?
    ImageNet uses specific naming conventions (e.g., "Granny_Smith")
    USDA uses natural language (e.g., "apple")
    
    This mapping improves search accuracy.
    """
    # Remove underscores and convert to lowercase
    cleaned = imagenet_name.replace('_', ' ').lower()
    
    # Common mappings
    mapping = {
        'granny smith': 'apple',
        'golden delicious': 'apple',
        'carbonara': 'pasta carbonara',
        'espresso': 'coffee',
        'french loaf': 'bread',
        'bagel': 'bagel',
        'pretzel': 'pretzel',
        'cheeseburger': 'cheeseburger',
        'hotdog': 'hot dog',
        'ice cream': 'ice cream',
        'pizza': 'pizza',
        'burrito': 'burrito',
        'guacamole': 'avocado',
    }
    
    return mapping.get(cleaned, cleaned)


# Testing code
if __name__ == "__main__":
    print("Testing USDA API integration...\n")
    print("Note: You need an API key to run this test")
    print("Get one at: https://fdc.nal.usda.gov/api-key-signup.html\n")
    
    # Initialize API (will use environment variable or prompt for key)
    try:
        api = USDANutritionAPI()
    except ValueError as e:
        print(f"❌ {e}")
        print("\nSet your API key:")
        print("  export USDA_API_KEY='your_key_here'  # macOS/Linux")
        print("  $env:USDA_API_KEY='your_key_here'     # Windows PowerShell")
        exit(1)
    
    # Test search
    print("Testing search for 'apple'...")
    results = api.search_food('apple', max_results=3)
    
    if results:
        print(f"\n✅ Found {len(results)} results:")
        for i, food in enumerate(results, 1):
            print(f"  {i}. {food['description']} (ID: {food['fdc_id']}, Type: {food['data_type']})")
    
    # Test detailed nutrition retrieval
    if results:
        print(f"\nRetrieving nutrition for: {results[0]['description']}")
        nutrition = api.get_nutrition_for_food('apple')
        
        if nutrition:
            print("\n✅ Nutritional Information (per 100g):")
            print(f"  Food: {nutrition['food_name']}")
            if 'calories' in nutrition:
                print(f"  Calories: {nutrition['calories']['amount']} {nutrition['calories']['unit']}")
            if 'protein' in nutrition:
                print(f"  Protein: {nutrition['protein']['amount']}g")
            if 'carbohydrates' in nutrition:
                print(f"  Carbs: {nutrition['carbohydrates']['amount']}g")
            if 'fat' in nutrition:
                print(f"  Fat: {nutrition['fat']['amount']}g")
    
    print("\n✅ API testing complete!")
