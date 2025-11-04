"""
Configuration Management

Centralized configuration prevents hardcoded values scattered throughout code.
Makes the application easier to maintain and deploy.
"""

import os
from typing import Dict, List

# ==================== API Configuration ====================

# USDA API settings
USDA_API_KEY = os.environ.get('USDA_API_KEY', '')
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
USDA_RATE_LIMIT = 0.1  # Seconds between requests

# ==================== Model Configuration ====================

# Available models with metadata
AVAILABLE_MODELS = {
    'MobileNet': {
        'size_mb': 16,
        'input_size': (224, 224),
        'speed': 'Fast',
        'accuracy': 'Good',
        'description': 'Optimized for mobile devices and real-time inference'
    },
    'ResNet50': {
        'size_mb': 98,
        'input_size': (224, 224),
        'speed': 'Medium',
        'accuracy': 'Better',
        'description': 'Balanced performance, widely used in production'
    },
    'InceptionV3': {
        'size_mb': 92,
        'input_size': (299, 299),
        'speed': 'Medium',
        'accuracy': 'Better',
        'description': 'Good for complex images with multiple objects'
    },
    'VGG16': {
        'size_mb': 528,
        'input_size': (224, 224),
        'speed': 'Slow',
        'accuracy': 'Good',
        'description': 'Simple architecture, large model size'
    },
    'DenseNet121': {
        'size_mb': 33,
        'input_size': (224, 224),
        'speed': 'Medium',
        'accuracy': 'Better',
        'description': 'Efficient parameter usage, good gradient flow'
    },
    'EfficientNetB0': {
        'size_mb': 29,
        'input_size': (224, 224),
        'speed': 'Fast',
        'accuracy': 'Best',
        'description': 'State-of-the-art efficiency, recommended for most use cases'
    },
    'Xception': {
        'size_mb': 88,
        'input_size': (299, 299),
        'speed': 'Medium',
        'accuracy': 'Best',
        'description': 'Depthwise separable convolutions, very deep features'
    }
}

# Default model for quick inference
DEFAULT_MODEL = 'EfficientNetB0'

# Ensemble model combination (for higher accuracy)
ENSEMBLE_MODELS = ['MobileNet', 'EfficientNetB0', 'ResNet50']

# ==================== Image Processing Configuration ====================

# Maximum image dimensions (for memory management)
MAX_IMAGE_DIMENSION = 1024  # Resize larger images before processing

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# ==================== UI Configuration ====================

# Streamlit page settings
PAGE_CONFIG = {
    'page_title': 'CalScan - Food Nutrition Analysis',
    'page_icon': '🍎',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Number of predictions to display
TOP_K_PREDICTIONS = 5

# Confidence threshold for displaying predictions (0-1)
CONFIDENCE_THRESHOLD = 0.01  # Show predictions with >1% confidence

# ==================== Food Category Mapping ====================

# ImageNet food-related class prefixes (WordNet IDs)
FOOD_WORDNET_PREFIXES = [
    'n07',  # food, nutrient
    'n12',  # plant  
    'n13',  # plant part
]

# Common ImageNet food classes (partial list for demonstration)
IMAGENET_FOOD_CLASSES = {
    'n07697313': 'cheeseburger',
    'n07697537': 'hotdog',
    'n07711569': 'mashed_potato',
    'n07714571': 'french_fries',
    'n07714990': 'broccoli',
    'n07715103': 'cauliflower',
    'n07716358': 'zucchini',
    'n07716906': 'spaghetti_squash',
    'n07717410': 'acorn_squash',
    'n07717556': 'butternut_squash',
    'n07718472': 'cucumber',
    'n07718747': 'artichoke',
    'n07720875': 'bell_pepper',
    'n07730033': 'cardoon',
    'n07734744': 'mushroom',
    'n07742313': 'Granny_Smith',
    'n07745940': 'strawberry',
    'n07747607': 'orange',
    'n07749582': 'lemon',
    'n07753113': 'fig',
    'n07753275': 'pineapple',
    'n07753592': 'banana',
    'n07754684': 'pomegranate',
    'n07760859': 'custard_apple',
    'n07768694': 'carbonara',
    'n07831146': 'croissant',
    'n07836838': 'chocolate_sauce',
    'n07860988': 'dough',
    'n07871810': 'bagel',
    'n07873807': 'pretzel',
    'n07875152': 'pizza',
    'n07880968': 'burrito',
    'n07892512': 'red_wine',
    'n07920052': 'espresso',
    'n07930864': 'cup',
    'n07932039': 'eggnog',
}

# Mapping problematic ImageNet names to USDA-friendly terms
IMAGENET_TO_USDA_MAPPING = {
    'granny_smith': 'apple, raw',
    'golden_delicious': 'apple, raw',
    'carbonara': 'pasta, carbonara',
    'espresso': 'coffee, brewed',
    'french_loaf': 'bread, french',
    'bagel': 'bagel, plain',
    'pretzel': 'pretzel',
    'cheeseburger': 'cheeseburger, single patty',
    'hotdog': 'hot dog, beef',
    'hot_dog': 'hot dog, beef',
    'ice_cream': 'ice cream, vanilla',
    'ice_lolly': 'popsicle',
    'pizza': 'pizza, cheese',
    'burrito': 'burrito, bean and cheese',
    'guacamole': 'avocado, raw',
    'mashed_potato': 'potato, mashed',
    'french_fries': 'potato, french fried',
    'spaghetti_squash': 'squash, winter, spaghetti',
    'acorn_squash': 'squash, winter, acorn',
    'butternut_squash': 'squash, winter, butternut',
    'bell_pepper': 'peppers, sweet, red',
    'red_wine': 'wine, red',
    'eggnog': 'eggnog',
}

# ==================== Nutrient Display Configuration ====================

# Nutrients to display (in order)
DISPLAY_NUTRIENTS = [
    'calories',
    'protein', 
    'carbohydrates',
    'fat',
    'fiber',
    'sugars',
    'sodium',
    'cholesterol'
]

# Nutrient display names and units
NUTRIENT_INFO = {
    'calories': {'name': 'Energy', 'unit': 'kcal', 'color': '#FF6B6B'},
    'protein': {'name': 'Protein', 'unit': 'g', 'color': '#4ECDC4'},
    'carbohydrates': {'name': 'Carbohydrates', 'unit': 'g', 'color': '#FFE66D'},
    'fat': {'name': 'Total Fat', 'unit': 'g', 'color': '#95E1D3'},
    'fiber': {'name': 'Dietary Fiber', 'unit': 'g', 'color': '#A8E6CF'},
    'sugars': {'name': 'Total Sugars', 'unit': 'g', 'color': '#FFB6B9'},
    'sodium': {'name': 'Sodium', 'unit': 'mg', 'color': '#C7CEEA'},
    'cholesterol': {'name': 'Cholesterol', 'unit': 'mg', 'color': '#FFDAB9'},
}

# ==================== Error Messages ====================

ERROR_MESSAGES = {
    'no_api_key': """
    ⚠️ USDA API Key Not Found
    
    To use nutritional data features, you need a free API key:
    1. Visit: https://fdc.nal.usda.gov/api-key-signup.html
    2. Sign up for a free account
    3. Set environment variable: USDA_API_KEY=your_key_here
    
    For now, you can still use the food classification features.
    """,
    
    'image_load_failed': """
    ❌ Failed to load image
    
    Please ensure:
    - File is a valid image format (JPG, PNG, BMP, GIF)
    - File is not corrupted
    - File size is reasonable (<10MB)
    """,
    
    'model_load_failed': """
    ❌ Failed to load model
    
    This might be due to:
    - Insufficient memory (try a smaller model like MobileNet)
    - TensorFlow installation issues
    - Network issues downloading weights
    """,
    
    'api_request_failed': """
    ❌ Failed to retrieve nutritional data
    
    Possible causes:
    - Network connectivity issues
    - API rate limit exceeded (1000 requests/hour)
    - Invalid API key
    """,
}

# ==================== Helper Functions ====================

def validate_config() -> Dict[str, bool]:
    """
    Validate configuration and return status.
    
    Returns:
        Dictionary with validation results for each component
    """
    status = {
        'api_key_set': bool(USDA_API_KEY),
        'tensorflow_available': False,
        'streamlit_available': False,
    }
    
    try:
        import tensorflow as tf
        status['tensorflow_available'] = True
        status['tensorflow_version'] = tf.__version__
    except ImportError:
        pass
    
    try:
        import streamlit as st
        status['streamlit_available'] = True
    except ImportError:
        pass
    
    return status


def get_model_info(model_name: str) -> Dict:
    """
    Get metadata for a specific model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model metadata
    """
    return AVAILABLE_MODELS.get(model_name, {})


def print_config_status():
    """Print configuration status for debugging."""
    status = validate_config()
    
    print("=" * 50)
    print("CalScan Configuration Status")
    print("=" * 50)
    
    print(f"\n✅ TensorFlow: {'Available' if status['tensorflow_available'] else '❌ Not installed'}")
    if status['tensorflow_available']:
        print(f"   Version: {status.get('tensorflow_version', 'Unknown')}")
    
    print(f"\n✅ Streamlit: {'Available' if status['streamlit_available'] else '❌ Not installed'}")
    
    print(f"\n✅ USDA API Key: {'Set' if status['api_key_set'] else '❌ Not set'}")
    if not status['api_key_set']:
        print("   Get key at: https://fdc.nal.usda.gov/api-key-signup.html")
    
    print(f"\n📊 Available Models: {len(AVAILABLE_MODELS)}")
    for name, info in AVAILABLE_MODELS.items():
        print(f"   - {name}: {info['description']}")
    
    print("\n" + "=" * 50)


# Testing code
if __name__ == "__main__":
    print_config_status()