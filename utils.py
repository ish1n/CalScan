"""
Utility Functions

Helper functions for common tasks across the application.
"""

import time
from functools import wraps
from typing import Callable, Any, Dict, List
import numpy as np
from PIL import Image
import io


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timer
        def slow_function():
            time.sleep(1)
    
    Why useful?
    - Identify performance bottlenecks
    - Compare model inference speeds
    - Monitor API response times
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️  {func.__name__} took {end - start:.3f}s")
        return result
    return wrapper


def format_confidence(probability: float) -> str:
    """
    Format probability as percentage string.
    
    Args:
        probability: Float between 0 and 1
    
    Returns:
        Formatted string like "85.32%"
    """
    return f"{probability * 100:.2f}%"


def format_nutrient_value(amount: float, unit: str) -> str:
    """
    Format nutrient amount with appropriate precision.
    
    Args:
        amount: Numeric value
        unit: Unit string (g, mg, kcal, etc.)
    
    Returns:
        Formatted string like "15.5g" or "250mg"
    """
    if amount >= 100:
        return f"{amount:.1f}{unit}"
    elif amount >= 10:
        return f"{amount:.1f}{unit}"
    else:
        return f"{amount:.2f}{unit}"


def resize_image_if_needed(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """
    Resize image if it exceeds maximum dimension.
    
    Why needed?
    - Large images consume excessive memory
    - Processing time increases with image size
    - Most detail is preserved even after resizing
    
    Args:
        image: PIL Image object
        max_dimension: Maximum width or height
    
    Returns:
        Resized image (or original if already small enough)
    """
    width, height = image.size
    
    if width <= max_dimension and height <= max_dimension:
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert PIL Image to bytes for Streamlit display.
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, etc.)
    
    Returns:
        Image as bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def calculate_macros_percentage(nutrients: Dict) -> Dict[str, float]:
    """
    Calculate percentage of calories from each macronutrient.
    
    Why useful?
    - Common dietary guideline: 40% carbs, 30% protein, 30% fat
    - Helps understand food composition beyond raw numbers
    
    Calorie conversions:
    - 1g protein = 4 kcal
    - 1g carbohydrates = 4 kcal
    - 1g fat = 9 kcal
    
    Args:
        nutrients: Dictionary with 'calories', 'protein', 'carbohydrates', 'fat'
    
    Returns:
        Dictionary with percentage of calories from each macro
    """
    if 'calories' not in nutrients or nutrients['calories']['amount'] == 0:
        return {}
    
    total_calories = nutrients['calories']['amount']
    
    percentages = {}
    
    if 'protein' in nutrients:
        protein_calories = nutrients['protein']['amount'] * 4
        percentages['protein'] = (protein_calories / total_calories) * 100
    
    if 'carbohydrates' in nutrients:
        carb_calories = nutrients['carbohydrates']['amount'] * 4
        percentages['carbohydrates'] = (carb_calories / total_calories) * 100
    
    if 'fat' in nutrients:
        fat_calories = nutrients['fat']['amount'] * 9
        percentages['fat'] = (fat_calories / total_calories) * 100
    
    return percentages


def filter_predictions_by_confidence(
    predictions: List[Dict],
    threshold: float = 0.01
) -> List[Dict]:
    """
    Filter predictions below confidence threshold.
    
    Args:
        predictions: List of prediction dictionaries
        threshold: Minimum confidence (0-1)
    
    Returns:
        Filtered list
    """
    return [p for p in predictions if p.get('probability', 0) >= threshold]


def get_color_for_nutrient(nutrient_name: str) -> str:
    """
    Get color code for nutrient visualization.
    
    Returns hex color code for consistent UI styling.
    """
    from config import NUTRIENT_INFO
    return NUTRIENT_INFO.get(nutrient_name, {}).get('color', '#808080')


def create_comparison_data(nutrients_list: List[Dict]) -> Dict:
    """
    Prepare data for side-by-side nutrient comparison.
    
    Useful when comparing multiple predictions or food alternatives.
    
    Args:
        nutrients_list: List of nutrient dictionaries
    
    Returns:
        Dictionary formatted for comparison display
    """
    if not nutrients_list:
        return {}
    
    from config import DISPLAY_NUTRIENTS
    
    comparison = {nutrient: [] for nutrient in DISPLAY_NUTRIENTS}
    food_names = []
    
    for nutrients in nutrients_list:
        food_names.append(nutrients.get('food_name', 'Unknown'))
        
        for nutrient in DISPLAY_NUTRIENTS:
            if nutrient in nutrients:
                amount = nutrients[nutrient]['amount']
            else:
                amount = 0
            comparison[nutrient].append(amount)
    
    comparison['food_names'] = food_names
    return comparison


def validate_image_file(file) -> tuple[bool, str]:
    """
    Validate uploaded image file.
    
    Args:
        file: Streamlit UploadedFile object
    
    Returns:
        (is_valid, error_message) tuple
    """
    from config import SUPPORTED_FORMATS
    
    if file is None:
        return False, "No file uploaded"
    
    # Check file extension
    file_ext = '.' + file.name.split('.')[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Use: {', '.join(SUPPORTED_FORMATS)}"
    
    # Check file size (10MB limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        return False, f"File too large. Maximum size: 10MB"
    
    # Try to open as image
    try:
        image = Image.open(file)
        image.verify()  # Verify it's a valid image
        file.seek(0)  # Reset for later use
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


# Testing code
if __name__ == "__main__":
    print("Testing utility functions...\n")
    
    # Test timer decorator
    @timer
    def slow_operation():
        time.sleep(0.5)
        return "Done"
    
    result = slow_operation()
    
    # Test formatting
    print(f"\nConfidence: {format_confidence(0.8532)}")
    print(f"Nutrient: {format_nutrient_value(15.7, 'g')}")
    
    # Test macro calculation
    sample_nutrients = {
        'calories': {'amount': 200, 'unit': 'kcal'},
        'protein': {'amount': 20, 'unit': 'g'},
        'carbohydrates': {'amount': 10, 'unit': 'g'},
        'fat': {'amount': 8, 'unit': 'g'}
    }
    
    macros = calculate_macros_percentage(sample_nutrients)
    print("\nMacro percentages:")
    for macro, percentage in macros.items():
        print(f"  {macro}: {percentage:.1f}%")
    
    print("\n✅ Utility testing complete!")