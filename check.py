import h5py

model_path = 'best_model.h5'  # Update if needed

with h5py.File(model_path, 'r') as f:
    keras_version = f.attrs.get('keras_version')
    backend = f.attrs.get('backend')
    print(f"Keras version used to save: {keras_version}")
    print(f"Backend: {backend}")
    if 'tensorflow_version' in f.attrs:
        print(f"TensorFlow version used to save: {f.attrs['tensorflow_version']}")
    else:
        print("TensorFlow version attribute not found in model file.")