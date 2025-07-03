import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import io
import matplotlib.pyplot as plt

class FloodImagePredictor:
    def __init__(self, model_path='best_model.h5', img_size=(128, 128)):
        self.model = load_model(model_path)
        self.img_size = img_size
    
    def load_tif_image(self, img_path):
        """Load and preprocess TIFF image"""
        try:
            img = io.imread(img_path)
            if len(img.shape) == 2:  # Grayscale
                img = np.stack((img,)*3, axis=-1)
            elif img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, self.img_size)
            return img / 255.0  # Normalize
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def predict_image(self, img_path):
        """Make prediction on single image"""
        img = self.load_tif_image(img_path)
        if img is not None:
            pred = self.model.predict(np.expand_dims(img, axis=0))[0][0]
            class_idx = 0 if pred < 0.5 else 1  # 0=Flood, 1=Non-Flood
            confidence = abs(pred-0.5)*200  # Convert to percentage
            return {
                'image': img,
                'class': 'Flood' if class_idx == 0 else 'Non-Flood',
                'confidence': confidence,
                'raw_prediction': pred
            }
        return None

def visualize_prediction(result, save_path='prediction_result.png'):
    """Display prediction results visually"""
    plt.figure(figsize=(8, 6))
    plt.imshow(result['image'])
    plt.title(f"Prediction: {result['class']}\nConfidence: {result['confidence']:.1f}%")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Result saved to {save_path}")
    print(f"Detailed prediction: {result}")

if __name__ == '__main__':
    # Initialize predictor
    predictor = FloodImagePredictor()
    
    # Path to your specific image
    image_path = r"C:\Users\sajib\OneDrive\Desktop\Flood_Prediction\data\train\images\train_images_flood\generate_AttentionGAN_hurricane-harvey_00000136_1.tif"
    
    # Make prediction
    result = predictor.predict_image(image_path)
    
    # Display results
    if result:
        visualize_prediction(result)
    else:
        print("Failed to process the image")