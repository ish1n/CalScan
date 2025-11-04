"""
Model Management Module

This module handles loading pre-trained models and performing inference.
We use models trained on ImageNet (1000 classes including many food items).
"""

import tensorflow as tf
from keras.applications import (
    MobileNet, ResNet50, InceptionV3, VGG16,
    DenseNet121, EfficientNetB0, Xception
)
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from typing import List, Tuple, Dict
import json

# Disable GPU if causing issues (optional)
# tf.config.set_visible_devices([], 'GPU')


class FoodClassifier:
    """
    Unified interface for multiple pre-trained models.
    
    Design Pattern: Strategy Pattern - allows switching between models easily
    """
    
    def __init__(self, model_name: str = 'MobileNet'):
        """
        Initialize classifier with specified model.
        
        Args:
            model_name: One of ['MobileNet', 'ResNet50', 'InceptionV3', 
                                 'VGG16', 'DenseNet121', 'EfficientNetB0', 'Xception']
        
        Why 'imagenet' weights?
        - ImageNet contains 1000 classes including ~300 food categories
        - Models trained on 1.2 million images over weeks of GPU time
        - Transfer learning leverages this pre-trained knowledge
        """
        self.model_name = model_name
        self.model = None
        self.input_size = self._get_input_size()
        self._load_model()
        
        print(f"✅ Loaded {model_name}")
        print(f"   Input size: {self.input_size}")
        print(f"   Parameters: {self.model.count_params():,}")
    
    def _get_input_size(self) -> Tuple[int, int]:
        """
        Get required input dimensions for each model.
        
        Why different sizes?
        - MobileNet: Designed for 224×224 (mobile efficiency)
        - InceptionV3/Xception: Use 299×299 (capture more detail)
        - Others: Standard 224×224
        """
        if self.model_name in ['InceptionV3', 'Xception']:
            return (299, 299)
        return (224, 224)
    
    def _load_model(self):
        """
        Load pre-trained model with ImageNet weights.
        
        include_top=True means we keep the classification head.
        This gives us predictions across all 1000 ImageNet classes.
        
        Alternative: include_top=False for custom classification layers
        (useful when fine-tuning for specific datasets)
        """
        model_map = {
            'MobileNet': MobileNet,
            'ResNet50': ResNet50,
            'InceptionV3': InceptionV3,
            'VGG16': VGG16,
            'DenseNet121': DenseNet121,
            'EfficientNetB0': EfficientNetB0,
            'Xception': Xception,
        }
        
        if self.model_name not in model_map:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model_class = model_map[self.model_name]
        
        # Load with pre-trained ImageNet weights
        self.model = model_class(
            weights='imagenet',  # Pre-trained weights
            include_top=True,     # Include classification layer
            input_shape=(*self.input_size, 3)  # (height, width, channels)
        )
        
        # Set to inference mode (disables training-specific layers like Dropout)
        self.model.trainable = False
    
    def predict(self, preprocessed_image: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Perform inference on preprocessed image.
        
        Args:
            preprocessed_image: Numpy array of shape (1, height, width, 3)
            top_k: Number of top predictions to return
        
        Returns:
            List of (class_id, class_name, probability) tuples
        
        How prediction works:
        1. Forward pass through network → logits (raw scores)
        2. Softmax activation → probabilities (sum to 1.0)
        3. decode_predictions → map class indices to human names
        """
        
        # Validate input
        if preprocessed_image.shape[1:3] != self.input_size:
            raise ValueError(
                f"Expected size {self.input_size}, got {preprocessed_image.shape[1:3]}"
            )
        
        # Forward pass - this is where the magic happens!
        # The image flows through convolutional layers extracting features
        predictions = self.model.predict(preprocessed_image, verbose=0)
        
        # Decode predictions from class indices to human-readable labels
        # decode_predictions returns: [[(class_id, class_name, probability), ...]]
        decoded = decode_predictions(predictions, top=top_k)[0]
        
        return decoded
    
    def predict_food_only(self, preprocessed_image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Filter predictions to food-related items only.
        
        ImageNet has ~300 food classes out of 1000 total.
        This filters out non-food predictions for better UX.
        """
        all_predictions = self.predict(preprocessed_image, top_k=10)
        
        # Food-related WordNet IDs (partial list - expand as needed)
        food_categories = {
            'n07': True,  # food, nutrient
            'n12': True,  # plant
            'n13': True,  # plant part
        }
        
        food_predictions = []
        for class_id, class_name, prob in all_predictions:
            # Check if class_id starts with food-related prefix
            if any(class_id.startswith(prefix) for prefix in food_categories):
                food_predictions.append({
                    'class_id': class_id,
                    'name': class_name,
                    'probability': float(prob),
                    'percentage': f"{prob * 100:.2f}%"
                })
            
            if len(food_predictions) >= top_k:
                break
        
        return food_predictions


class EnsembleClassifier:
    """
    Combine multiple models for more robust predictions.
    
    Why ensemble?
    - Different models learn different features
    - Averaging reduces individual model errors
    - Improved confidence estimation
    
    Trade-off: Slower inference (runs multiple models)
    """
    
    def __init__(self, model_names: List[str] = None):
        if model_names is None:
            # Default: fast + accurate combination
            model_names = ['MobileNet', 'EfficientNetB0', 'ResNet50']
        
        self.classifiers = [FoodClassifier(name) for name in model_names]
        print(f"\n✅ Ensemble loaded with {len(self.classifiers)} models")
    
    def predict(self, preprocessed_images: Dict[str, np.ndarray], top_k: int = 5) -> List[Dict]:
        """
        Aggregate predictions from multiple models.
        
        Args:
            preprocessed_images: Dict mapping model_name → preprocessed_image
            top_k: Number of top predictions to return
        
        Strategy: Average probabilities across models
        """
        # Collect all predictions
        all_class_scores = {}
        
        for classifier in self.classifiers:
            img = preprocessed_images.get(classifier.model_name)
            if img is None:
                continue
            
            predictions = classifier.predict(img, top_k=10)
            
            for class_id, class_name, prob in predictions:
                if class_id not in all_class_scores:
                    all_class_scores[class_id] = {
                        'name': class_name,
                        'scores': []
                    }
                all_class_scores[class_id]['scores'].append(prob)
        
        # Average scores
        averaged_predictions = []
        for class_id, data in all_class_scores.items():
            avg_score = np.mean(data['scores'])
            averaged_predictions.append({
                'class_id': class_id,
                'name': data['name'],
                'probability': float(avg_score),
                'percentage': f"{avg_score * 100:.2f}%",
                'num_models': len(data['scores'])
            })
        
        # Sort by probability
        averaged_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return averaged_predictions[:top_k]


# Testing code
if __name__ == "__main__":
    print("Testing model loading and inference...\n")
    
    # Test single model
    classifier = FoodClassifier('MobileNet')
    
    # Create dummy image for testing
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_image = dummy_image * 2 - 1  # Scale to [-1, 1] for MobileNet
    
    predictions = classifier.predict(dummy_image, top_k=3)
    print("\nSample predictions (random image):")
    for class_id, name, prob in predictions:
        print(f"  {name}: {prob*100:.2f}%")
    
    print("\n✅ Model testing complete!")