import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def model_diagnosis():
    """Test model with synthetic images"""
    try:
        model = load_model('best_model.h5')
    except:
        print("No trained model found. Please train first.")
        return
    
    # Create test cases
    test_cases = {
        'Black': np.zeros((128, 128, 3)),
        'White': np.ones((128, 128, 3)),
        'Gray': np.full((128, 128, 3), 0.5),
        'Random': np.random.rand(128, 128, 3)
    }
    
    # Predict and display
    plt.figure(figsize=(15, 10))
    for i, (name, img) in enumerate(test_cases.items()):
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        class_label = 'Flood' if pred < 0.5 else 'Non-Flood'
        confidence = abs(pred - 0.5) * 200  # 0-100% scale
        
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.title(f"{name}\n{class_label} ({confidence:.1f}% conf)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_diagnosis.png')
    plt.show()

if __name__ == '__main__':
    model_diagnosis()