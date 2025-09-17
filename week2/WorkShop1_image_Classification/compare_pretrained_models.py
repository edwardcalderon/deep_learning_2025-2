"""
Comparison of Pre-trained Models for Cat and Dog Classification

This script compares the performance of three pre-trained models (EfficientNetB0, ResNet50, and VGG16)
on the Cats vs Dogs dataset. It also includes functionality to test the best model with downloaded images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    EfficientNetB0, ResNet50, VGG16,
    efficientnet, resnet, vgg16
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import requests
from PIL import Image
from io import BytesIO

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2
TRAIN_DIR = './dataset_dogs_vs_cats/train'
TEST_DIR = './dataset_dogs_vs_cats/test'
MODEL_SAVE_DIR = './saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Data preprocessing
def create_data_generators():
    """Create data generators for training and validation."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def create_model(base_model, input_shape, model_name):
    """Create a model based on a pre-trained base model."""
    # Create the base model
    if model_name == 'efficientnet':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif model_name == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif model_name == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=input_shape)
    
    # Preprocess input based on model requirements
    if model_name == 'efficientnet':
        x = efficientnet.preprocess_input(inputs)
    elif model_name == 'resnet50':
        x = resnet.preprocess_input(inputs)
    elif model_name == 'vgg16':
        x = vgg16.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    return model

def train_and_evaluate_models():
    """Train and evaluate multiple pre-trained models."""
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Define models to train
    models = {
        'efficientnet': (EfficientNetB0, (224, 224, 3)),
        'resnet50': (ResNet50, (224, 224, 3)),
        'vgg16': (VGG16, (224, 224, 3))
    }
    
    results = {}
    
    for model_name, (base_model_class, input_shape) in models.items():
        print(f"\nTraining {model_name}...")
        
        # Create and compile model
        model = create_model(base_model_class, input_shape, model_name)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.h5"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        # Train the model
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(test_gen)
        print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
        
        # Save results
        results[model_name] = {
            'model': model,
            'history': history.history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        }
    
    return results, test_gen

def plot_training_history(history, model_name):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.show()

def test_with_downloaded_images(model, class_indices):
    """Test the model with downloaded images from the internet."""
    # Example image URLs (replace with actual cat and dog image URLs)
    test_images = {
        'cat1': 'https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_1280.jpg',
        'cat2': 'https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554_1280.jpg',
        'dog1': 'https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg',
        'dog2': 'https://cdn.pixabay.com/photo/2018/01/09/11/04/dog-3071334_1280.jpg',
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, url) in enumerate(test_images.items(), 1):
        try:
            # Download and preprocess image
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array)[0][0]
            predicted_class = 'dog' if prediction > 0.5 else 'cat'
            confidence = prediction if predicted_class == 'dog' else (1 - prediction)
            
            # Plot image with prediction
            plt.subplot(2, 2, i)
            plt.imshow(img)
            plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}')
            plt.axis('off')
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    plt.tight_layout()
    plt.savefig('predictions_on_downloaded_images.png')
    plt.show()

def main():
    # Train and evaluate models
    results, test_gen = train_and_evaluate_models()
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with test accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    
    # Plot training history for all models
    for model_name, result in results.items():
        plot_training_history(result['history'], model_name)
    
    # Test with downloaded images using the best model
    print("\nTesting with downloaded images...")
    test_with_downloaded_images(best_model, test_gen.class_indices)
    
    # Save the best model
    best_model.save(os.path.join(MODEL_SAVE_DIR, 'best_model.h5'))
    print(f"\nBest model saved to {os.path.join(MODEL_SAVE_DIR, 'best_model.h5')}")

if __name__ == "__main__":
    main()
