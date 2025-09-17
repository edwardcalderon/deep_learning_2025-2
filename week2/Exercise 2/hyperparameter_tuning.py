import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def resolve_dataset_path() -> str:
    """Try multiple likely dataset paths and return the first that exists."""
    here = os.path.dirname(__file__)
    candidates = [
        # Week1 workshop dataset observed in the workspace
        os.path.normpath(os.path.join(here, "..", "..", "week1", "workshop0", "housepricedata.csv")),
        # In case the script is run from project root
        os.path.normpath(os.path.join(here, "..", "week1", "workshop0", "housepricedata.csv")),
        # Current directory fallback
        os.path.join(here, "housepricedata.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate dataset. Expected one of: " + " | ".join(candidates))

def load_and_preprocess_data():
    """Load and preprocess dataset. Automatically infers regression/classification."""
    data_path = resolve_dataset_path()
    data = pd.read_csv(data_path)

    # Infer target column
    target_col = None
    task_type = None  # 'regression' or 'classification'
    if 'price' in data.columns:
        target_col = 'price'
        task_type = 'regression'
    elif 'AboveMedianPrice' in data.columns:
        target_col = 'AboveMedianPrice'
        task_type = 'classification'
    else:
        # If not found, assume last column is target
        target_col = data.columns[-1]
        # Binary classification if values are 0/1
        unique_vals = pd.unique(data[target_col])
        if set(np.unique(unique_vals)).issubset({0, 1}):
            task_type = 'classification'
        else:
            task_type = 'regression'

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, task_type, X.shape[1]

def build_model(hp, n_features: int, task_type: str):
    """Build the model with hyperparameters to tune."""
    model = keras.Sequential()
    
    # Tune the number of units in the first Dense layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation='relu', input_shape=(n_features,)))
    
    # Tune the number of hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))

    # Output layer depends on task type
    if task_type == 'classification':
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(1))  # regression
    
    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    if task_type == 'classification':
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    return model

def run_tuning(X_train_scaled, y_train, X_test_scaled, y_test, task_type: str, n_features: int, trials=5):
    """Run hyperparameter tuning with the specified number of trials."""
    def model_builder(hp):
        return build_model(hp, n_features=n_features, task_type=task_type)

    # Choose objective based on task
    if task_type == 'classification':
        objective = 'val_accuracy'
    else:
        objective = 'val_mae'

    tuner = kt.RandomSearch(
        model_builder,
        objective=objective,
        max_trials=trials,
        directory='tuning',
        project_name=f'house_price_trials_{trials}_{task_type}'
    )
    
    # Display search space
    tuner.search_space_summary()
    
    # Perform the hyperparameter search
    tuner.search(
        X_train_scaled,
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    # Train the model
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # Evaluate the model
    eval_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    if task_type == 'classification':
        # Keras returns [loss, accuracy, auc]
        test_metrics = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'auc': float(eval_results[2]) if len(eval_results) > 2 else None,
        }
    else:
        # Keras returns [loss, mae]
        test_metrics = {
            'loss': float(eval_results[0]),
            'mae': float(eval_results[1]),
        }
    
    return model, history, test_metrics, best_hps

def plot_training_history(history, title):
    """Plot training and validation loss and MAE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    
    # Plot MAE
    if 'mae' in history.history:
        ax2.plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history:
        ax2.plot(history.history['val_mae'], label='Validation MAE')
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'{title} - Secondary Metric')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, task_type, n_features = load_and_preprocess_data()
    
    # Dictionary to store results
    results = {}
    
    # Run tuning with different numbers of trials
    for trials in [3, 5, 8]:
        print(f"\n{'='*50}")
        print(f"Running {trials} trials")
        print(f"{'='*50}")
        
        model, history, test_metrics, best_hps = run_tuning(
            X_train_scaled, y_train, X_test_scaled, y_test, task_type, n_features, trials
        )
        
        # Store results
        results[trials] = {
            'test_metrics': test_metrics,
            'best_hps': best_hps,
            'history': history
        }
        
        # Plot training history
        plot_training_history(history, f'Training History ({trials} Trials)')
    
    # Print summary of results
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    rows = []
    for trials, result in results.items():
        print(f"\nResults for {trials} trials:")
        metrics = result['test_metrics']
        if 'accuracy' in metrics:
            print(f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics.get('auc', float('nan')):.4f}, Loss: {metrics['loss']:.4f}")
        else:
            print(f"MAE: {metrics['mae']:.2f}, Loss: {metrics['loss']:.2f}")
        print("Best Hyperparameters:")
        for param, value in result['best_hps'].values.items():
            print(f"  {param}: {value}")
        row = {'trials': trials, **metrics}
        rows.append(row)

    # Create and save summary table
    summary_df = pd.DataFrame(rows).sort_values('trials')
    out_path = os.path.join(os.path.dirname(__file__), f"tuning_summary_{task_type}.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary table to: {out_path}")

if __name__ == "__main__":
    main()
