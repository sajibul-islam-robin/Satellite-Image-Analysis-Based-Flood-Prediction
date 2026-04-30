import tensorflow as tf
from data_processing import DataProcessor
from model import FloodModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
def plot_training_history(history):
    """Plot training and validation metrics"""
    metrics = ['accuracy', 'loss', 'precision', 'recall', 'auc']
    
    plt.figure(figsize=(18, 12))
    
    for idx, metric in enumerate(metrics):
        plt.subplot(3, 2, idx+1)
        
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Train {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            plt.title(f'{metric.capitalize()} Curve')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
        else:
            print(f"Warning: {metric} not found in history.")
    
    plt.tight_layout()
    plt.savefig('model_curve.png')
    plt.close()


def train_model():
    # Initialize
    processor = DataProcessor('data')
    processor.verify_data_distribution()
    
    # Load data
    (train_images, train_labels), (val_images, val_labels), _ = processor.prepare_datasets()
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    print(f"\nClass weights: {class_weights}")
    
    # Build model
    flood_model = FloodModel()
    model = flood_model.build_model()
    model = flood_model.compile_model(model)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_auc',
            mode='max',
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Train
    history = model.fit(
        train_images[:, :, :, :3],  # Only image channels
        train_labels,
        validation_data=(val_images[:, :, :, :3], val_labels),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training curves
    plot_training_history(history)
    
    return model, history

if __name__ == '__main__':
    model, history = train_model()