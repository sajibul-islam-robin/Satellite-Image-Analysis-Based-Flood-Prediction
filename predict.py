import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def predict_single_image(image_path):
    """Make prediction on a single image"""
    try:
        model = load_model('best_model.h5')
    except:
        print("Error: Model not found. Train first or check path.")
        return None
    
    # Load and preprocess image
    try:
        img = io.imread(image_path)
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Predict
    pred = model.predict(np.expand_dims(img, axis=0))[0][0]
    class_label = 'Flood' if pred < 0.5 else 'Non-Flood'
    confidence = abs(pred - 0.5) * 200
    
    # Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {class_label}\nConfidence: {confidence:.1f}%")
    plt.axis('off')
    plt.savefig('prediction_result.png')
    plt.show()
    
    return {
        'class': class_label,
        'confidence': confidence,
        'raw_prediction': float(pred)
    }

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = predict_single_image(sys.argv[1])
        print("\nPrediction Result:")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Raw Output: {result['raw_prediction']:.4f}")
    else:
        print("Usage: python predict.py <image_path>")