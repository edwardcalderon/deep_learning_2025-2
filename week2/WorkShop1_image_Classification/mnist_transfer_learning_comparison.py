"""
MNIST Classification with Transfer Learning Comparison

This script extends the original MNIST classification work by:
1. Using the trained LeNet-5 model for predictions on external images
2. Comparing different pre-trained architectures using Transfer Learning
3. Evaluating performance across MobileNet, VGG16, EfficientNet, and ResNet
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import MobileNet, VGG16, EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
import cv2
import time
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_prepare_mnist():
    """Load and prepare MNIST data"""
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)

def create_lenet5_model():
    """Create LeNet-5 model (from original notebook)"""
    print("Creating LeNet-5 model...")
    model = Sequential()
    
    # CONV1 and MAX-POOLING1
    model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    # CONV2 and MAX-POOLING2
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    # Flatten, FC1, FC2 and output
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    
    # Compile model
    model.compile(optimizer=SGD(learning_rate=0.01), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    return model

def train_lenet5(model, x_train, y_train_cat, x_test, y_test_cat):
    """Train LeNet-5 model"""
    print("Training LeNet-5 model...")
    history = model.fit(x_train, y_train_cat, 
                       epochs=5, 
                       batch_size=128, 
                       validation_data=(x_test, y_test_cat),
                       verbose=1)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"LeNet-5 Test Accuracy: {test_acc:.4f}")
    
    return history, test_acc

def get_local_test_images():
    """Get local test images (0.jpeg and 1.jpeg)"""
    print("Loading local test images...")
    test_images = []
    
    # List of local test images with their true labels
    local_images = [
        ("0.jpeg", 0),  # First digit should be 0
        ("1.jpeg", 1)   # Second digit should be 1
    ]
    
    for img_file, true_label in local_images:
        img_path = os.path.join(os.getcwd(), img_file)
        if os.path.exists(img_path):
            test_images.append((img_path, true_label))
            print(f"Found test image: {img_file} (True label: {true_label})")
        else:
            print(f"Warning: Test image not found: {img_file}")
    
    if not test_images:
        print("No test images found. Please make sure 0.jpeg and 1.jpeg are in the current directory.")
    else:
        print(f"Loaded {len(test_images)} test images")
    
    return test_images

def preprocess_external_image(image_path, target_size=(28, 28), for_transfer_learning=False):
    """Preprocess external image to match MNIST format or transfer learning input"""
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply thresholding to get binary image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours and get the largest one (the digit)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box of the digit
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Extract the digit
        digit = img[y:y+h, x:x+w]
        
        # Calculate the size for the square image (max of width and height)
        max_dim = max(w, h)
        
        # Create a square black image
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        # Calculate position to center the digit
        x_offset = (max_dim - w) // 2
        y_offset = (max_dim - h) // 2
        
        # Place the digit in the center of the square
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        
        # Resize to target size
        img = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)
    else:
        # If no contours found, just resize the original image
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # For transfer learning, convert to RGB and resize to 32x32
    if for_transfer_learning:
        # Convert to 3 channels (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Resize to 32x32
        img_rgb = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)
        # Normalize
        img_rgb = img_rgb.astype('float32') / 255.0
        return img_rgb.reshape(1, 32, 32, 3)
    
    # For LeNet-5, keep as grayscale and 28x28
    img = img.astype('float32') / 255.0
    return img.reshape(1, 28, 28, 1)

def predict_external_images(models, test_images):
    """Make predictions on external images using all trained models"""
    print("Making predictions on external images...\n")
    
    all_predictions = {}
    
    for model_name, model_info in models.items():
        if 'model' not in model_info:
            continue
            
        print(f"\n{'='*60}")
        print(f"Predictions using {model_name}")
        print("="*60)
        
        model = model_info['model']
        predictions = []
        
        for img_path, true_label in test_images:
            # Preprocess image based on model type
            is_transfer = model_name != 'LeNet-5'
            target_size = (32, 32) if is_transfer else (28, 28)
            processed_img = preprocess_external_image(img_path, target_size, is_transfer)
            
            # Make prediction
            pred = model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(pred)
            confidence = np.max(pred) * 100  # as percentage
            
            predictions.append({
                'image_path': os.path.basename(img_path),
                'true_label': true_label,
                'predicted_digit': predicted_digit,
                'confidence': confidence
            })
            
            # Display results
            print(f"Image: {os.path.basename(img_path)}")
            print(f"True label: {true_label}, Predicted: {predicted_digit}, Confidence: {confidence:.2f}%")
            
            # Display the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            plt.figure(figsize=(4, 4))
            plt.imshow(img, cmap='gray')
            plt.title(f"{os.path.basename(img_path)} (True: {true_label}, Pred: {predicted_digit})")
            plt.axis('off')
            plt.show()
            
            print("-" * 60)
        
        all_predictions[model_name] = predictions
    
    return all_predictions

def create_transfer_learning_model(base_model_name, input_shape=(32, 32, 3)):
    """Create transfer learning model with specified base architecture"""
    print(f"Creating {base_model_name} transfer learning model...")
    
    # Define base model with pre-trained weights
    base_model = None
    
    try:
        if base_model_name == 'MobileNet':
            base_model = MobileNet(weights='imagenet', 
                                 include_top=False, 
                                 input_shape=input_shape)
        elif base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', 
                             include_top=False, 
                             input_shape=input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet',
                                include_top=False,
                                input_shape=input_shape)
        else:
            raise ValueError(f"Unsupported model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(10, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model with lower learning rate
        model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model
        
    except Exception as e:
        print(f"Error creating {base_model_name} model: {str(e)}")
        return None

def prepare_data_for_transfer_learning(x_train, x_test):
    """Prepare data for transfer learning (convert to RGB)"""
    # Convert grayscale to RGB by repeating the channel
    x_train_rgb = np.repeat(x_train, 3, axis=-1)
    x_test_rgb = np.repeat(x_test, 3, axis=-1)
    
    # Resize to minimum input size for pre-trained models (32x32)
    x_train_resized = tf.image.resize(x_train_rgb, [32, 32]).numpy()
    x_test_resized = tf.image.resize(x_test_rgb, [32, 32]).numpy()
    
    return x_train_resized, x_test_resized

def train_transfer_learning_models(x_train, y_train_cat, x_test, y_test_cat):
    """Train all transfer learning models and compare results"""
    print("Training transfer learning models...")
    
    # Prepare data for transfer learning
    x_train_rgb, x_test_rgb = prepare_data_for_transfer_learning(x_train, x_test)
    
    # Only use models that work well with MNIST
    models = ['MobileNet', 'VGG16', 'ResNet50']
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name}")
        print('='*60)
        
        try:
            # Create model
            model = create_transfer_learning_model(model_name, input_shape=(32, 32, 3))
            if model is None:
                raise Exception(f"Failed to create {model_name} model")
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=2,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            start_time = time.time()
            history = model.fit(
                x_train_rgb, y_train_cat,
                epochs=10,  # Max epochs, will stop early if no improvement
                batch_size=64,
                validation_data=(x_test_rgb, y_test_cat),
                callbacks=callbacks,
                verbose=1
            )
            training_time = time.time() - start_time
            
            # Evaluate model
            test_loss, test_acc = model.evaluate(x_test_rgb, y_test_cat, verbose=0)
            
            results[model_name] = {
                'test_accuracy': test_acc,
                'training_time': training_time,
                'model': model,
                'history': history.history
            }
            
            print(f"\n{model_name} Results:")
            print(f"- Test Accuracy: {test_acc:.4f}")
            print(f"- Training Time: {training_time/60:.2f} minutes")
            print(f"- Parameters: {model.count_params():,}")
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title(f'{model_name} Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title(f'{model_name} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = {
                'test_accuracy': 0.0,
                'training_time': 0.0,
                'error': str(e)
            }
    
    return results

def compare_results(lenet_accuracy, transfer_results):
    """Compare results and draw conclusions"""
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"LeNet-5 (Original):     {lenet_accuracy:.4f}")
    print("-" * 40)
    
    # Sort transfer learning results by accuracy
    sorted_results = sorted([(name, results) for name, results in transfer_results.items() 
                           if 'test_accuracy' in results], 
                          key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    for model_name, results in sorted_results:
        if 'error' not in results:
            print(f"{model_name:20}: {results['test_accuracy']:.4f} (Time: {results['training_time']:.1f}s)")
        else:
            print(f"{model_name:20}: Failed - {results['error']}")
    
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    
    if sorted_results:
        best_model = sorted_results[0]
        print(f"1. Best performing model: {best_model[0]} with {best_model[1]['test_accuracy']:.4f} accuracy")
        
        if best_model[1]['test_accuracy'] > lenet_accuracy:
            print(f"2. Transfer learning outperformed LeNet-5 by {best_model[1]['test_accuracy'] - lenet_accuracy:.4f}")
        else:
            print(f"2. LeNet-5 outperformed transfer learning by {lenet_accuracy - best_model[1]['test_accuracy']:.4f}")
        
        print("3. Transfer learning observations:")
        for model_name, results in sorted_results:
            if 'error' not in results:
                if 'MobileNet' in model_name:
                    print("   - MobileNet: Lightweight, good for mobile deployment")
                elif 'VGG16' in model_name:
                    print("   - VGG16: Simple architecture, good baseline")
                elif 'EfficientNet' in model_name:
                    print("   - EfficientNet: Balanced efficiency and accuracy")
                elif 'ResNet' in model_name:
                    print("   - ResNet: Deep architecture with skip connections")
        
        print("4. For MNIST digit classification:")
        print("   - Simple architectures like LeNet-5 are often sufficient")
        print("   - Transfer learning may be overkill for this simple task")
        print("   - Pre-trained models excel more on complex, natural image tasks")

def main():
    """Main execution function"""
    print("MNIST Classification with Transfer Learning Comparison")
    print("="*70)
    
    # Load and prepare data
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_and_prepare_mnist()
    
    # Dictionary to store all models and their results
    all_models = {}
    
    # Create and train LeNet-5 model
    print("\n" + "="*70)
    print("TRAINING LENET-5 MODEL")
    print("="*70)
    lenet_model = create_lenet5_model()
    lenet_history, lenet_accuracy = train_lenet5(lenet_model, x_train, y_train_cat, x_test, y_test_cat)
    all_models['LeNet-5'] = {
        'model': lenet_model,
        'test_accuracy': lenet_accuracy,
        'history': lenet_history
    }
    
    # Train transfer learning models
    print("\n" + "="*70)
    print("TRAINING TRANSFER LEARNING MODELS")
    print("="*70)
    transfer_results = train_transfer_learning_models(x_train, y_train_cat, x_test, y_test_cat)
    all_models.update(transfer_results)
    
    # Get local test images
    print("\n" + "="*70)
    print("TESTING ON LOCAL IMAGES")
    print("="*70)
    test_images = get_local_test_images()
    
    if test_images:
        # Make predictions on test images using all models
        all_predictions = predict_external_images(all_models, test_images)
    
    # Compare results and draw conclusions
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    compare_results(lenet_accuracy, transfer_results)
    
    print("\nAnalysis complete!")
    
    # Display final comparison of all models
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<15} {'Test Accuracy':<15} {'Parameters'}")
    print("-" * 40)
    
    # Add LeNet-5 to results for comparison
    transfer_results['LeNet-5'] = {
        'test_accuracy': lenet_accuracy,
        'training_time': 0  # Not tracked for LeNet-5 in this example
    }
    
    # Sort by accuracy
    sorted_results = sorted(transfer_results.items(), 
                          key=lambda x: x[1]['test_accuracy'], 
                          reverse=True)
    
    for model_name, results in sorted_results:
        if 'error' in results:
            print(f"{model_name:<15} {'Error':<15} {results['error']}")
        else:
            param_count = results.get('model').count_params() if 'model' in results else "N/A"
            print(f"{model_name:<15} {results['test_accuracy']:<15.4f} {param_count:,}")
    
    print("\nNote: Lower parameter count with high accuracy is generally better.")

if __name__ == "__main__":
    main()
