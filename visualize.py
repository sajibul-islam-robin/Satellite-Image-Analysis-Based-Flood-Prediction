import tensorflow as tf
from tensorflow.keras.utils import plot_model
from model import FloodModel  # Import your existing model class
import matplotlib.pyplot as plt
from PIL import Image

def generate_model_diagram():
    """Generate and display model architecture diagram"""
    # Initialize and build model
    flood_model = FloodModel()
    model = flood_model.build_model()
    
    # Generate standard Keras plot
    plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to Bottom layout
        dpi=150,       # Higher resolution
        expand_nested=False
    )
    
    # Generate layered view (alternative visualization)
    try:
        import visualkeras
        font = ImageFont.truetype("arial.ttf", 14) if tf.sys.platform == 'win32' else None
        visualkeras.layered_view(
            model,
            to_file='model_layered.png',
            legend=True,
            scale_xy=20,
            font=font,
            spacing=30
        )
    except ImportError:
        print("visualkeras not installed, skipping layered view")
    
    # Print model summary
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    model.summary()
    
    # Display the generated image
    try:
        img = Image.open('model_architecture.png')
        plt.figure(figsize=(12, 16))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Model Architecture', pad=20)
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("Could not display model image")

def main():
    print("Generating model visualization...")
    generate_model_diagram()
    print("\nVisualization saved to:")
    print("- model_architecture.png (standard Keras diagram)")
    print("- model_layered.png (alternative view)")

if __name__ == '__main__':
    main()