"""
CalScan - Food Nutrition Analysis Application

Main Streamlit application integrating computer vision and nutritional data.
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Import our modules
from models import FoodClassifier, EnsembleClassifier
from preprocessing import load_and_preprocess_image, preprocess_for_model
from nutrition_api import USDANutritionAPI, map_imagenet_to_usda
from config import (
    PAGE_CONFIG, AVAILABLE_MODELS, DEFAULT_MODEL, TOP_K_PREDICTIONS,
    CONFIDENCE_THRESHOLD, ERROR_MESSAGES, DISPLAY_NUTRIENTS, NUTRIENT_INFO,
    USDA_API_KEY
)
from utils import (
    timer, format_confidence, format_nutrient_value,
    resize_image_if_needed, calculate_macros_percentage,
    validate_image_file
)

# ==================== Page Configuration ====================

st.set_page_config(**PAGE_CONFIG)

# ==================== Caching Functions ====================

@st.cache_resource
def load_model(model_name: str) -> FoodClassifier:
    """
    Load and cache model to avoid reloading on each interaction.
    
    Why @st.cache_resource?
    - Models are large (16MB - 528MB)
    - Loading takes 2-5 seconds
    - Models don't change during app lifetime
    - This decorator caches the model in memory
    
    The cached model persists across all users and sessions.
    """
    with st.spinner(f'Loading {model_name}...'):
        return FoodClassifier(model_name)


@st.cache_resource
def load_api_client() -> USDANutritionAPI:
    """Cache API client to reuse HTTP session."""
    if not USDA_API_KEY:
        return None
    return USDANutritionAPI(USDA_API_KEY)


# ==================== UI Components ====================

def render_header():
    """Render application header and description."""
    st.title("🍎 CalScan")
    st.markdown("""
    **Automated Food Nutrition Analysis** using Computer Vision and Machine Learning
    
    Upload a food image to:
    1. 🔍 Identify the food using deep learning
    2. 📊 Retrieve detailed nutritional information
    3. 💡 Get insights about your food
    """)
    st.divider()


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Selection")
        
        use_ensemble = st.checkbox(
            "Use Ensemble (3 models)",
            value=False,
            help="Combines multiple models for higher accuracy but slower inference"
        )
        
        if not use_ensemble:
            selected_model = st.selectbox(
                "Choose Model",
                options=list(AVAILABLE_MODELS.keys()),
                index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
                help="Different models offer speed/accuracy tradeoffs"
            )
            
            # Display model info
            model_info = AVAILABLE_MODELS[selected_model]
            st.info(f"""
            **{selected_model}**
            - Size: {model_info['size_mb']}MB
            - Speed: {model_info['speed']}
            - Accuracy: {model_info['accuracy']}
            
            {model_info['description']}
            """)
        else:
            selected_model = "Ensemble"
            st.info("""
            **Ensemble Mode**
            Combines predictions from:
            - MobileNet (fast)
            - EfficientNetB0 (accurate)
            - ResNet50 (robust)
            
            ⚠️ Takes 3x longer than single model
            """)
        
        st.divider()
        
        # Prediction settings
        st.subheader("Prediction Settings")
        
        top_k = st.slider(
            "Number of predictions",
            min_value=1,
            max_value=10,
            value=TOP_K_PREDICTIONS,
            help="How many top predictions to display"
        )
        
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=0.5,
            value=CONFIDENCE_THRESHOLD,
            step=0.01,
            format="%.2f",
            help="Minimum confidence to display a prediction"
        )
        
        st.divider()
        
        # API status
        st.subheader("API Status")
        if USDA_API_KEY:
            st.success("✅ USDA API Connected")
        else:
            st.warning("⚠️ USDA API Not Configured")
            with st.expander("How to get API key"):
                st.markdown(ERROR_MESSAGES['no_api_key'])
        
        return selected_model, top_k, confidence_threshold, use_ensemble


def render_image_upload():
    """Render image upload widget and return uploaded image."""
    st.subheader("📤 Upload Food Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        help="Upload a clear image of a single food item for best results"
    )
    
    if uploaded_file is not None:
        # Validate image
        is_valid, error_msg = validate_image_file(uploaded_file)
        
        if not is_valid:
            st.error(error_msg)
            return None
        
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Resize if too large (for display and processing)
        image = resize_image_if_needed(image, max_dimension=800)
        
        # Display image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        return image
    
    return None


def render_predictions(predictions: list, api_client: USDANutritionAPI):
    """
    Render classification predictions and nutritional data.
    
    Args:
        predictions: List of prediction dictionaries
        api_client: USDA API client (or None if not configured)
    """
    st.subheader("🔍 Classification Results")
    
    if not predictions:
        st.warning("No predictions above confidence threshold")
        return
    
    # Display predictions in expandable sections
    for i, pred in enumerate(predictions, 1):
        confidence = pred.get('probability', 0)
        food_name = pred.get('name','Unknown')
        
        # Create expander for each prediction
        with st.expander(
            f"#{i}: {food_name.replace('_', ' ').title()} - {format_confidence(confidence)}",
            expanded=(i == 1)  # Expand first prediction by default
        ):
            # Display confidence with progress bar
            st.markdown(f"**Confidence:** {format_confidence(confidence)}")
            st.progress(float(confidence))
            
            # If API is available, fetch nutrition data
            if api_client and i == 1:  # Only fetch for top prediction
                st.divider()
                st.markdown("### 📊 Nutritional Information")
                
                with st.spinner("Fetching nutrition data..."):
                    # Map ImageNet name to USDA-friendly term
                    usda_query = map_imagenet_to_usda(food_name)
                    nutrition = api_client.get_nutrition_for_food(usda_query)
                
                if nutrition:
                    render_nutrition_data(nutrition)
                else:
                    st.warning(f"Could not find nutritional data for '{usda_query}'")
                    st.info("Try searching for a more specific food name or check the USDA database manually.")
            
            elif api_client and i > 1:
                st.info("💡 Nutritional data shown for top prediction only. Select this prediction to see details.")
            
            else:
                st.info("🔑 Configure USDA API key in sidebar to view nutritional data")


def render_nutrition_data(nutrition: dict):
    """
    Render nutritional information in a user-friendly format.
    
    Args:
        nutrition: Dictionary with nutritional data from USDA API
    """
    if not nutrition:
        return
    
    # Display food name and serving size
    food_name = nutrition.get('food_name', 'Unknown Food')
    serving_size = nutrition.get('serving_size', 100)
    serving_unit = nutrition.get('serving_unit', 'g')
    
    st.markdown(f"**Food:** {food_name}")
    st.markdown(f"**Serving Size:** {serving_size}{serving_unit}")
    
    st.divider()
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Macros", "📋 Details"])
    
    with tab1:
        render_nutrition_overview(nutrition)
    
    with tab2:
        render_macros_breakdown(nutrition)
    
    with tab3:
        render_detailed_nutrients(nutrition)


def render_nutrition_overview(nutrition: dict):
    """Render key nutritional highlights."""
    
    # Display key nutrients in columns
    col1, col2, col3, col4 = st.columns(4)
    
    key_nutrients = ['calories', 'protein', 'carbohydrates', 'fat']
    cols = [col1, col2, col3, col4]
    
    for nutrient, col in zip(key_nutrients, cols):
        if nutrient in nutrition:
            amount = nutrition[nutrient]['amount']
            unit = nutrition[nutrient]['unit']
            name = NUTRIENT_INFO[nutrient]['name']
            color = NUTRIENT_INFO[nutrient]['color']
            
            with col:
                st.markdown(f"""
                <div style="
                    background-color: {color}20;
                    border-left: 4px solid {color};
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                ">
                    <h3 style="margin: 0; color: {color};">{amount}{unit}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">{name}</p>
                </div>
                """, unsafe_allow_html=True)


def render_macros_breakdown(nutrition: dict):
    """Render macronutrient breakdown with visualization."""
    
    # Calculate macro percentages
    macro_percentages = calculate_macros_percentage(nutrition)
    
    if not macro_percentages:
        st.info("Insufficient data to calculate macronutrient breakdown")
        return
    
    st.markdown("### Calories from Macronutrients")
    
    # Display as columns with progress bars
    for macro in ['protein', 'carbohydrates', 'fat']:
        if macro in macro_percentages:
            percentage = macro_percentages[macro]
            name = NUTRIENT_INFO[macro]['name']
            color = NUTRIENT_INFO[macro]['color']
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{name}**")
                st.progress(percentage / 100)
            
            with col2:
                st.markdown(f"**{percentage:.1f}%**")
    
    st.divider()
    
    # Display recommended ranges
    st.markdown("### Dietary Reference")
    st.info("""
    **Typical Macronutrient Ranges:**
    - Protein: 10-35% of calories
    - Carbohydrates: 45-65% of calories
    - Fat: 20-35% of calories
    
    *Note: Individual needs vary based on activity level, health goals, and metabolic factors.*
    """)


def render_detailed_nutrients(nutrition: dict):
    """Render detailed list of all available nutrients."""
    
    st.markdown("### Complete Nutrient Profile (per 100g)")
    
    # Create a table of nutrients
    for nutrient_key in DISPLAY_NUTRIENTS:
        if nutrient_key in nutrition:
            nutrient_data = nutrition[nutrient_key]
            amount = nutrient_data['amount']
            unit = nutrient_data['unit']
            name = NUTRIENT_INFO.get(nutrient_key, {}).get('name', nutrient_key.title())
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{name}**")
            
            with col2:
                st.markdown(f"`{format_nutrient_value(amount, unit)}`")
    
    st.divider()
    
    # Add disclaimer
    st.caption("""
    ℹ️ Nutritional values are approximate and may vary based on:
    - Preparation method
    - Brand or variety
    - Ripeness or freshness
    - Cooking techniques
    
    Data source: USDA FoodData Central
    """)


def render_about():
    """Render about section with project information."""
    with st.expander("ℹ️ About CalScan"):
        st.markdown("""
        ### How It Works
        
        **CalScan** uses state-of-the-art deep learning models to identify foods from images,
        then retrieves nutritional information from the USDA database.
        
        #### Technology Stack
        - **Computer Vision**: TensorFlow/Keras with pre-trained models (ImageNet)
        - **Models Available**: MobileNet, ResNet50, InceptionV3, VGG16, DenseNet121, EfficientNetB0, Xception
        - **Nutrition Data**: USDA FoodData Central API (300,000+ foods)
        - **Interface**: Streamlit web framework
        
        #### Model Comparison
        
        | Model | Best For |
        |-------|----------|
        | MobileNet | Mobile devices, real-time |
        | EfficientNetB0 | Best accuracy/efficiency balance ⭐ |
        | ResNet50 | Production applications |
        | InceptionV3 | Complex multi-object images |
        | Xception | Maximum accuracy |
        | Ensemble | Highest confidence (slower) |
        
        #### Limitations
        - Works best with single, clearly visible food items
        - Accuracy depends on image quality and lighting
        - Some foods may not be in ImageNet training data
        - Nutritional values are estimates (vary by preparation)
        
        #### Tips for Best Results
        1. 📸 Use clear, well-lit images
        2. 🎯 Center the food item in frame
        3. 📏 Show the food from a typical viewing angle
        4. 🚫 Avoid cluttered backgrounds
        5. 🔍 Single food items work better than mixed dishes
        
        ---
        
        **Made with ❤️ using TensorFlow and Streamlit**
        """)


# ==================== Main Application Flow ====================

def main():
    """Main application entry point."""
    
    # Render header
    render_header()
    
    # Render sidebar and get settings
    selected_model, top_k, confidence_threshold, use_ensemble = render_sidebar()
    
    # Initialize API client
    api_client = load_api_client()
    
    # Render image upload
    image = render_image_upload()
    
    # If image is uploaded, process it
    if image is not None:
        st.divider()
        
        # Add process button
        if st.button("🔬 Analyze Food", type="primary", use_container_width=True):
            
            # Create progress bar
            progress_bar = st.progress(0, text="Starting analysis...")
            
            try:
                # Step 1: Load model(s)
                progress_bar.progress(20, text="Loading model...")
                
                if use_ensemble:
                    # Load ensemble models
                    from config import ENSEMBLE_MODELS
                    classifiers = {
                        name: load_model(name) for name in ENSEMBLE_MODELS
                    }
                    
                    # Preprocess for each model
                    progress_bar.progress(40, text="Preprocessing image...")
                    preprocessed_images = {}
                    
                    for name, classifier in classifiers.items():
                        img_array = load_and_preprocess_image(
                            image,
                            target_size=classifier.input_size
                        )
                        preprocessed_images[name] = preprocess_for_model(img_array, name)
                    
                    # Run ensemble prediction
                    progress_bar.progress(60, text="Running ensemble prediction...")
                    
                    ensemble = EnsembleClassifier(list(classifiers.keys()))
                    predictions = ensemble.predict(preprocessed_images, top_k=top_k)
                    
                else:
                    # Single model prediction
                    classifier = load_model(selected_model)
                    
                    # Preprocess image
                    progress_bar.progress(40, text="Preprocessing image...")
                    img_array = load_and_preprocess_image(
                        image,
                        target_size=classifier.input_size
                    )
                    preprocessed = preprocess_for_model(img_array, selected_model)
                    
                    # Run prediction
                    progress_bar.progress(60, text="Classifying food...")
                    raw_predictions = classifier.predict(preprocessed, top_k=top_k)
                    
                    # Format predictions
                    predictions = [{
                        'class_id': class_id,
                        'name': name,
                        'probability': float(prob),
                        'percentage': format_confidence(prob)
                    } for class_id, name, prob in raw_predictions]
                
                # Filter by confidence threshold
                predictions = [
                    p for p in predictions 
                    if p['probability'] >= confidence_threshold
                ]
                
                # Complete progress
                progress_bar.progress(100, text="Analysis complete!")
                
                # Clear progress bar after a moment
                import time
                time.sleep(0.5)
                progress_bar.empty()
                
                # Display results
                st.success("✅ Analysis complete!")
                render_predictions(predictions, api_client)
                
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)  # Show full traceback in development
    
    # Render about section
    st.divider()
    render_about()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>CalScan v1.0 | Built with TensorFlow, Streamlit & USDA FoodData Central</p>
        <p style="font-size: 0.8em;">⚠️ For educational purposes. Consult professionals for dietary advice.</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== Entry Point ====================

if __name__ == "__main__":
    main()