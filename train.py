import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import cv2

class GPDSDataset:
    def __init__(self, gpds_path, img_size=(128, 128), max_persons=None):
        self.gpds_path = gpds_path
        self.img_size = img_size
        self.pairs = []
        self.labels = []
        
        # Get all person IDs from the filenames
        person_ids = set()
        
        # Check genuine folder for person IDs
        genuine_path = os.path.join(gpds_path, "genuine")
        if os.path.exists(genuine_path):
            for img_file in os.listdir(genuine_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and img_file.startswith('c-'):
                    try:
                        person_id = int(img_file.split('-')[1])
                        person_ids.add(person_id)
                    except (ValueError, IndexError):
                        continue
        
        # Check forge folder for person IDs
        forge_path = os.path.join(gpds_path, "forge")
        if os.path.exists(forge_path):
            for img_file in os.listdir(forge_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and img_file.startswith('cf-'):
                    try:
                        person_id = int(img_file.split('-')[1])
                        person_ids.add(person_id)
                    except (ValueError, IndexError):
                        continue
        
        person_dirs = sorted(list(person_ids))
        if max_persons:
            person_dirs = person_dirs[:max_persons]
        
        print(f"Processing {len(person_dirs)} persons...")
        
        # Load all images and create pairs
        self._load_data(person_dirs)
        
    def _load_data(self, person_dirs):
        all_genuine = []
        all_forge = []
        
        # Load genuine signatures from genuine folder
        genuine_path = os.path.join(self.gpds_path, "genuine")
        if os.path.exists(genuine_path):
            for img_file in os.listdir(genuine_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract person ID from filename: c-XXX-YY -> XXX
                    if img_file.startswith('c-'):
                        try:
                            person_id = int(img_file.split('-')[1])
                            if person_id in person_dirs:
                                img_path = os.path.join(genuine_path, img_file)
                                all_genuine.append((img_path, person_id))
                        except (ValueError, IndexError):
                            continue
        
        # Load forge signatures from forge folder
        forge_path = os.path.join(self.gpds_path, "forge")
        if os.path.exists(forge_path):
            for img_file in os.listdir(forge_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract person ID from filename: cf-XXX-YY -> XXX
                    if img_file.startswith('cf-'):
                        try:
                            person_id = int(img_file.split('-')[1])
                            if person_id in person_dirs:
                                img_path = os.path.join(forge_path, img_file)
                                all_forge.append((img_path, person_id))
                        except (ValueError, IndexError):
                            continue
        
        print(f"Loaded {len(all_genuine)} genuine and {len(all_forge)} forge signatures")
        
        # Create positive pairs (genuine-genuine from same person)
        genuine_by_person = {}
        for img_path, person_id in all_genuine:
            if person_id not in genuine_by_person:
                genuine_by_person[person_id] = []
            genuine_by_person[person_id].append(img_path)
        
        for person_id, images in genuine_by_person.items():
            # Create pairs within same person
            for i in range(len(images)):
                for j in range(i+1, min(i+4, len(images))):  # Limit pairs per image
                    self.pairs.append((images[i], images[j]))
                    self.labels.append(1)  # Same person (genuine)
        
        # Create negative pairs (genuine-forge)
        for genuine_img, genuine_person in all_genuine:
            # Select forge signatures (some from same person, some from different)
            forge_candidates = []
            
            # Add forge from same person (skilled forgeries)
            same_person_forges = [img for img, person in all_forge if person == genuine_person]
            forge_candidates.extend(random.sample(same_person_forges, min(2, len(same_person_forges))))
            
            # Add forge from different persons (random forgeries)
            diff_person_forges = [img for img, person in all_forge if person != genuine_person]
            forge_candidates.extend(random.sample(diff_person_forges, min(3, len(diff_person_forges))))
            
            for forge_img in forge_candidates:
                self.pairs.append((genuine_img, forge_img))
                self.labels.append(0)  # Different (forge)
        
        print(f"Created {len(self.pairs)} pairs:")
        print(f"  - Genuine pairs: {sum(self.labels)}")
        print(f"  - Forge pairs: {len(self.labels) - sum(self.labels)}")
    
    def preprocess_image(self, img_path):
        """Load and preprocess image"""
        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Try with PIL if cv2 fails
                img = Image.open(img_path).convert('L')
                img = np.array(img)
            
            # Resize
            img = cv2.resize(img, self.img_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add channel dimension
            img = np.expand_dims(img, axis=-1)
            
            return img
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return np.zeros((*self.img_size, 1), dtype=np.float32)
    
    def get_data(self, test_size=0.2):
        """Generate training and validation data"""
        # Split pairs
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            self.pairs, self.labels, test_size=test_size, random_state=42, stratify=self.labels
        )
        
        # Load and preprocess images
        def load_pair_data(pairs, labels):
            img1_list = []
            img2_list = []
            label_list = []
            
            for (img1_path, img2_path), label in zip(pairs, labels):
                img1 = self.preprocess_image(img1_path)
                img2 = self.preprocess_image(img2_path)
                
                img1_list.append(img1)
                img2_list.append(img2)
                label_list.append(label)
            
            return np.array(img1_list), np.array(img2_list), np.array(label_list)
        
        print("Loading training data...")
        train_img1, train_img2, train_labels = load_pair_data(train_pairs, train_labels)
        
        print("Loading validation data...")
        val_img1, val_img2, val_labels = load_pair_data(val_pairs, val_labels)
        
        return (train_img1, train_img2, train_labels), (val_img1, val_img2, val_labels)

def create_siamese_network(input_shape=(128, 128, 1)):
    """Create Siamese network using Keras functional API"""
    
    # Define the base network (feature extractor)
    input_layer = layers.Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(96, (11, 11), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Create the base model
    base_network = keras.Model(input_layer, x, name='base_network')
    
    # Create Siamese network
    input_a = layers.Input(shape=input_shape, name='input_a')
    input_b = layers.Input(shape=input_shape, name='input_b')
    
    # Get feature vectors
    feature_a = base_network(input_a)
    feature_b = base_network(input_b)
    
    # Calculate L1 distance
    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([feature_a, feature_b])
    
    # Final prediction
    prediction = layers.Dense(1, activation='sigmoid', name='prediction')(distance)
    
    # Create and compile model
    siamese_model = keras.Model(inputs=[input_a, input_b], outputs=prediction)
    
    return siamese_model

def train_model(model, train_data, val_data, epochs=50, batch_size=32):
    """Train the Siamese network"""
    
    train_img1, train_img2, train_labels = train_data
    val_img1, val_img2, val_labels = val_data
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_siamese.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        [train_img1, train_img2], train_labels,
        validation_data=([val_img1, val_img2], val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def predict_signature(model, ref_img_path, test_img_path, img_size=(128, 128)):
    """Predict if two signatures are from the same person"""
    
    def preprocess_single_image(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = Image.open(img_path).convert('L')
            img = np.array(img)
        
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    
    ref_img = preprocess_single_image(ref_img_path)
    test_img = preprocess_single_image(test_img_path)
    
    prediction = model.predict([ref_img, test_img])[0][0]
    return prediction

def main():
    # Configuration
    GPDS_PATH = "gpds/train"
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 50
    MAX_PERSONS = None  # Limit for faster testing, set to None for all
    
    print("Loading GPDS dataset...")
    dataset = GPDSDataset(GPDS_PATH, IMG_SIZE, MAX_PERSONS)
    
    print("Preparing data...")
    train_data, val_data = dataset.get_data(test_size=0.2)
    
    print("Creating Siamese network...")
    model = create_siamese_network(input_shape=(*IMG_SIZE, 1))
    model.summary()
    
    print("Training model...")
    history = train_model(model, train_data, val_data, EPOCHS, BATCH_SIZE)
    
    # Save final model
    model.save('final_siamese.keras')
    print("Model saved as 'final_siamese.keras'")
    
    # Evaluate on validation set
    val_img1, val_img2, val_labels = val_data
    val_loss, val_accuracy = model.evaluate([val_img1, val_img2], val_labels, verbose=0)
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    
if __name__ == "__main__":
    main()
