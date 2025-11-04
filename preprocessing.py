"""
Image Preprocessing Module

This module handles converting raw images into model-ready tensors.
Different models require different preprocessing strategies.
"""

from PIL import Image
import numpy as np
import tensorflow as tf
from typing import Tuple

from PIL import Image
import numpy as np
import tensorflow as tf
from typing import Tuple


def load_and_preprocess_image(
    image_path_or_file, 
    target_size: Tuple[int, int] = (224, 224),
    preprocessing_function=None
) -> np.ndarray:
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path_or_file: File path string, file-like object, or PIL Image instance
        target_size: (height, width) tuple - model's expected input size
        preprocessing_function: Model-specific preprocessing (e.g., from tf.keras.applications)
    
    Returns:
        Preprocessed image as numpy array with shape (1, height, width, 3)
    """

    try:
        # Fix: Check if input is already a PIL Image
        if isinstance(image_path_or_file, Image.Image):
            img = image_path_or_file
        elif isinstance(image_path_or_file, str):
            img = Image.open(image_path_or_file)
        else:
            img = Image.open(image_path_or_file)
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(target_size, Image.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    if preprocessing_function:
        img_array = preprocessing_function(img_array)
    else:
        img_array = img_array / 255.0
    
    return img_array



def validate_image(image_array: np.ndarray) -> bool:
    """
    Validate preprocessed image meets requirements.
    
    Common issues:
    - Wrong number of dimensions (should be 4: batch, height, width, channels)
    - Wrong data type (should be float32)
    - Out of range values (depends on preprocessing, but typically [-1, 1] or [0, 1])
    """
    if image_array.ndim != 4:
        print(f"❌ Wrong dimensions: {image_array.ndim} (expected 4)")
        return False
    
    if image_array.shape[0] != 1:
        print(f"❌ Wrong batch size: {image_array.shape[0]} (expected 1)")
        return False
    
    if image_array.shape[-1] != 3:
        print(f"❌ Wrong channels: {image_array.shape[-1]} (expected 3 for RGB)")
        return False
    
    if image_array.dtype != np.float32:
        print(f"⚠️ Suboptimal dtype: {image_array.dtype} (float32 recommended)")
    
    print(f"✅ Image validated: shape={image_array.shape}, dtype={image_array.dtype}")
    return True


def preprocess_for_model(image_array: np.ndarray, model_name: str) -> np.ndarray:
    """
    Apply model-specific preprocessing transformations.
    
    Why different models need different preprocessing:
    - MobileNet: Trained with pixels scaled to [-1, 1]
    - ResNet50: Uses "caffe" mode (BGR channel order, ImageNet mean subtraction)
    - InceptionV3: Scaled to [-1, 1]
    - VGG16: Uses "caffe" mode with different mean values
    
    These differences come from how the original models were trained.
    Using the wrong preprocessing will drastically reduce accuracy!
    """
    
    from tensorflow.keras.applications import (
        mobilenet, resnet50, inception_v3, vgg16, 
        densenet, efficientnet, xception
    )
    
    preprocessing_map = {
        'MobileNet': mobilenet.preprocess_input,
        'ResNet50': resnet50.preprocess_input,
        'InceptionV3': inception_v3.preprocess_input,
        'VGG16': vgg16.preprocess_input,
        'DenseNet121': densenet.preprocess_input,
        'EfficientNetB0': efficientnet.preprocess_input,
        'Xception': xception.preprocess_input,
    }
    
    if model_name in preprocessing_map:
        return preprocessing_map[model_name](image_array.copy())
    else:
        # Fallback: scale to [0, 1]
        return image_array / 255.0 if image_array.max() > 1.0 else image_array


# Testing code
if __name__ == "__main__":
    print("Testing image preprocessing pipeline...\n")
    
    # Test with a sample image (you'll need to provide one)
    # For now, create a dummy image
    from PIL import Image
    import tempfile
    
    # Create test image
    test_img = Image.new('RGB', (500, 500), color='red')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        test_img.save(f.name)
        test_path = f.name
    
    # Test preprocessing
    img_array = load_and_preprocess_image(test_path, target_size=(224, 224))
    validate_image(img_array)
    
    print(f"\nPixel value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    print(f"Mean: {img_array.mean():.3f}")