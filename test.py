import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage import io
import math

class FloodImageTester:
    def __init__(self, model_path='best_model.h5', img_size=(128, 128)):
        self.model = load_model(model_path)
        self.img_size = img_size
        self.class_names = ['Flood', 'Non-Flood']
    
    def load_tif_image(self, img_path):
        """Load and preprocess TIFF image"""
        try:
            img = io.imread(img_path)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, self.img_size)
            return img / 255.0
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def predict_images(self, test_dir):
        """Predict on all images in test directory"""
        results = []
        for img_name in os.listdir(test_dir):
            if img_name.lower().endswith('.tif'):
                img_path = os.path.join(test_dir, img_name)
                img = self.load_tif_image(img_path)
                if img is not None:
                    pred = self.model.predict(np.expand_dims(img, axis=0))[0][0]
                    class_idx = 0 if pred < 0.5 else 1
                    confidence = abs(pred-0.5)*200  # Convert to 0-100% scale
                    results.append({
                        'image': img,
                        'prediction': class_idx,
                        'confidence': confidence,
                        'name': img_name
                    })
        return results
    
    def visualize_results(self, results, save_path='test_results.png'):
        """Create visualization of prediction results"""
        num_images = len(results)
        total_plots = num_images + 1  # +1 for summary
        cols = 3
        rows = math.ceil(total_plots / cols)

        plt.figure(figsize=(5 * cols, 4 * rows))
        for i, result in enumerate(results):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(result['image'])
            plt.title(f"{result['name']}\n"
                      f"Pred: {self.class_names[result['prediction']]}\n"
                      f"Conf: {result['confidence']:.1f}%")
            plt.axis('off')

        # Add summary statistics
        flood_count = sum(1 for r in results if r['prediction'] == 0)
        avg_conf = np.mean([r['confidence'] for r in results])

        plt.subplot(rows, cols, total_plots)
        plt.text(0.1, 0.6,
                 f"Total Images: {num_images}\n"
                 f"Flood Predictions: {flood_count}\n"
                 f"Non-Flood Predictions: {num_images - flood_count}\n"
                 f"Avg Confidence: {avg_conf:.1f}%",
                 fontsize=12)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Results saved to {save_path}")

if __name__ == '__main__':
    # Initialize tester
    tester = FloodImageTester()
    
    # Path to your test flood images
    test_dir = r'C:\Users\sajib\OneDrive\Desktop\Flood_Prediction\data\test\images\test_images_flood'
    
    # Make predictions
    results = tester.predict_images(test_dir)
    
    # Visualize results
    if results:
        tester.visualize_results(results)
    else:
        print("No valid TIFF images found in the test directory")