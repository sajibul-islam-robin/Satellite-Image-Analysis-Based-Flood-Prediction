import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from skimage import io
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, data_path, img_size=(256, 256)):
        self.data_path = data_path
        self.img_size = img_size
        self.classes = ['flood', 'non_flood']
        
    def load_tif_image(self, path):
        """Special handling for TIFF images"""
        try:
            img = io.imread(path)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
        
    def load_dataset(self, dataset_type='train'):
        """Load images and masks for train/val/test"""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            image_dir = os.path.join(self.data_path, dataset_type, 'images', f'{dataset_type}_images_{class_name}')
            mask_dir = os.path.join(self.data_path, dataset_type, 'masks', f'{dataset_type}_masks_{class_name}')
            
            for img_name in os.listdir(image_dir):
                if img_name.lower().endswith('.tif'):
                    img_path = os.path.join(image_dir, img_name)
                    img = self.load_tif_image(img_path)
                    if img is None:
                        continue
                        
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    
                    # Load mask (assuming .png masks)
                    mask_name = img_name.replace('.tif', '.png')
                    mask_path = os.path.join(mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, self.img_size)
                        mask = mask / 255.0
                        mask = np.expand_dims(mask, axis=-1)
                    else:
                        mask = np.zeros((*self.img_size, 1))
                    
                    combined = np.concatenate([img, mask], axis=-1)
                    images.append(combined)
                    labels.append(class_idx)
        
        return shuffle(np.array(images), np.array(labels), random_state=42)
    
    def verify_data_distribution(self):
        """Verify class distribution across datasets"""
        print("\n=== DATA DISTRIBUTION ===")
        for dataset in ['train', 'val', 'test']:
            try:
                images, labels = self.load_dataset(dataset)
                print(f"\n{dataset.upper()} SET:")
                print(f"Flood images: {np.sum(labels==0)}")
                print(f"Non-flood images: {np.sum(labels==1)}")
                print(f"Ratio: {np.sum(labels==0)/len(labels):.2f}:{np.sum(labels==1)/len(labels):.2f}")
                
                # Visualize samples
                flood_idx = np.where(labels==0)[0][0]
                non_flood_idx = np.where(labels==1)[0][0]
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(images[flood_idx][:, :, :3])
                plt.title(f"Sample Flood ({dataset})")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(images[non_flood_idx][:, :, :3])
                plt.title(f"Sample Non-Flood ({dataset})")
                plt.axis('off')
                
                plt.savefig(f'{dataset}_samples.png')
                plt.close()
                
            except Exception as e:
                print(f"Could not analyze {dataset} set: {str(e)}")
    
    def prepare_datasets(self):
        """Prepare all datasets"""
        train_data = self.load_dataset('train')
        val_data = self.load_dataset('val')
        test_images, test_labels = self.load_test_data()
        return train_data, val_data, (test_images, test_labels)
    
    def load_test_data(self):
        """Load test images without masks"""
        test_images = []
        test_labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            image_dir = os.path.join(self.data_path, 'test', 'images', f'test_images_{class_name}')
            
            for img_name in os.listdir(image_dir):
                if img_name.lower().endswith('.tif'):
                    img_path = os.path.join(image_dir, img_name)
                    img = self.load_tif_image(img_path)
                    if img is None:
                        continue
                        
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    test_images.append(img)
                    test_labels.append(class_idx)
        
        return np.array(test_images), np.array(test_labels)