import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

class FloodModel:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
    
    def build_model(self):
        """Build balanced CNN architecture"""
        model = models.Sequential([
            # Entry block
            layers.Conv2D(32, (3,3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001),
                        input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.2),
            
            # Middle block
            layers.Conv2D(64, (3,3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.3),
            
            # Exit block
            layers.Conv2D(128, (3,3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Classifier
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def compile_model(self, model):
        """Compile with appropriate metrics"""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        return model