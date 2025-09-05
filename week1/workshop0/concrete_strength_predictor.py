"""
Predictor de Resistencia del Concreto
Utiliza el dataset Concrete_Data_Yeh.csv y una red neuronal para predecir
la resistencia a la compresión del concreto con valores aleatorios de prueba.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Configuración de semilla para reproducibilidad
np.random.seed(42)

def load_and_prepare_data():
    """Carga y prepara los datos del dataset de concreto"""
    print("Cargando dataset de concreto...")
    
    # Cargar el dataset
    data = pd.read_csv('Concrete_Data_Yeh.csv')
    
    print(f"Dataset cargado: {data.shape[0]} muestras, {data.shape[1]} características")
    print("\nCaracterísticas del dataset:")
    print(data.columns.tolist())
    print("\nEstadísticas descriptivas:")
    print(data.describe())
    
    # Separar características y objetivo
    X = data.drop("csMPa", axis=1).values
    y = data["csMPa"].values
    
    return X, y, data.columns[:-1].tolist()

def create_and_train_model(X_train, y_train, X_val, y_val):
    """Crea y entrena el modelo de red neuronal"""
    print("\nCreando modelo de red neuronal...")
    
    # Arquitectura del modelo (basada en el notebook original)
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Salida de regresión
    ])
    
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='mse', 
                 metrics=['mae'])
    
    print("Arquitectura del modelo:")
    model.summary()
    
    # Entrenar el modelo
    print("\nEntrenando modelo...")
    history = model.fit(X_train, y_train, 
                       validation_data=(X_val, y_val),
                       epochs=100, 
                       batch_size=32, 
                       verbose=1)
    
    return model, history

def generate_random_samples(feature_names, data_stats, n_samples=5):
    """Genera muestras aleatorias basadas en las estadísticas del dataset"""
    print(f"\nGenerando {n_samples} muestras aleatorias para predicción...")
    
    random_samples = []
    
    # Rangos típicos basados en el análisis del dataset
    ranges = {
        'cement': (100, 540),
        'slag': (0, 360),
        'flyash': (0, 200),
        'water': (120, 220),
        'superplasticizer': (0, 35),
        'coarseaggregate': (800, 1150),
        'fineaggregate': (600, 950),
        'age': (1, 365)
    }
    
    for i in range(n_samples):
        sample = []
        print(f"\nMuestra {i+1}:")
        for feature in feature_names:
            if feature in ranges:
                min_val, max_val = ranges[feature]
                value = np.random.uniform(min_val, max_val)
            else:
                # Usar estadísticas del dataset si no hay rango definido
                mean = data_stats[feature]['mean']
                std = data_stats[feature]['std']
                value = np.random.normal(mean, std)
                value = max(0, value)  # Asegurar valores no negativos
            
            sample.append(value)
            print(f"  {feature}: {value:.2f}")
        
        random_samples.append(sample)
    
    return np.array(random_samples)

def evaluate_predictions(model, scaler, random_samples, feature_names):
    """Evalúa las predicciones del modelo con las muestras aleatorias"""
    print("\n" + "="*60)
    print("RESULTADOS DE PREDICCIÓN")
    print("="*60)
    
    # Normalizar las muestras aleatorias
    random_samples_scaled = scaler.transform(random_samples)
    
    # Hacer predicciones
    predictions = model.predict(random_samples_scaled, verbose=0)
    
    results = []
    
    for i, (sample, pred) in enumerate(zip(random_samples, predictions)):
        print(f"\nMUESTRA {i+1}:")
        print("-" * 40)
        
        # Mostrar valores de entrada
        print("Valores de entrada:")
        for j, (feature, value) in enumerate(zip(feature_names, sample)):
            print(f"  {feature:20s}: {value:8.2f}")
        
        # Mostrar predicción
        pred_value = pred[0]
        print(f"\nRESISTencia PREDICHA: {pred_value:.2f} MPa")
        
        # Clasificar la resistencia
        if pred_value < 20:
            classification = "Baja resistencia"
        elif pred_value < 40:
            classification = "Resistencia media"
        elif pred_value < 60:
            classification = "Alta resistencia"
        else:
            classification = "Muy alta resistencia"
        
        print(f"Clasificación: {classification}")
        
        results.append({
            'sample_id': i+1,
            'inputs': dict(zip(feature_names, sample)),
            'predicted_strength': pred_value,
            'classification': classification
        })
    
    return results

def calculate_model_performance(model, X_test, y_test):
    """Calcula métricas de rendimiento del modelo"""
    print("\n" + "="*60)
    print("RENDIMIENTO DEL MODELO")
    print("="*60)
    
    # Predicciones en el conjunto de prueba
    y_pred = model.predict(X_test, verbose=0)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Error Absoluto Medio (MAE):     {mae:.2f} MPa")
    print(f"Error Cuadrático Medio (MSE):   {mse:.2f}")
    print(f"Raíz del MSE (RMSE):           {rmse:.2f} MPa")
    print(f"Coeficiente de Determinación (R²): {r2:.4f}")
    
    # Calcular error porcentual
    mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
    print(f"Error Porcentual Absoluto Medio: {mape:.2f}%")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def plot_results(history, y_test, y_pred):
    """Genera gráficos de los resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico 1: Pérdida durante entrenamiento
    axes[0, 0].plot(history.history['loss'], label='Entrenamiento')
    axes[0, 0].plot(history.history['val_loss'], label='Validación')
    axes[0, 0].set_title('Pérdida del Modelo')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Gráfico 2: MAE durante entrenamiento
    axes[0, 1].plot(history.history['mae'], label='Entrenamiento')
    axes[0, 1].plot(history.history['val_mae'], label='Validación')
    axes[0, 1].set_title('Error Absoluto Medio')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Gráfico 3: Predicho vs Real
    axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1, 0].set_xlabel('Resistencia Real (MPa)')
    axes[1, 0].set_ylabel('Resistencia Predicha (MPa)')
    axes[1, 0].set_title('Predicho vs Real')
    axes[1, 0].grid(True)
    
    # Gráfico 4: Distribución de errores
    errors = y_test - y_pred.flatten()
    axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Error (Real - Predicho)')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Errores')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('concrete_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Función principal"""
    print("PREDICTOR DE RESISTENCIA DEL CONCRETO")
    print("="*60)
    
    # 1. Cargar y preparar datos
    X, y, feature_names = load_and_prepare_data()
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 3. Normalizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Crear y entrenar modelo
    model, history = create_and_train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 5. Evaluar rendimiento del modelo
    y_pred = model.predict(X_test_scaled, verbose=0)
    performance = calculate_model_performance(model, X_test_scaled, y_test)
    
    # 6. Generar muestras aleatorias
    data_df = pd.DataFrame(X, columns=feature_names)
    data_stats = data_df.describe().to_dict()
    random_samples = generate_random_samples(feature_names, data_stats, n_samples=5)
    
    # 7. Hacer predicciones con muestras aleatorias
    results = evaluate_predictions(model, scaler, random_samples, feature_names)
    
    # 8. Generar gráficos
    plot_results(history, y_test, y_pred)
    
    # 9. Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Modelo entrenado con {len(X_train)} muestras")
    print(f"Evaluado en {len(X_test)} muestras de prueba")
    print(f"Error promedio: {performance['mae']:.2f} MPa")
    print(f"Precisión del modelo (R²): {performance['r2']:.4f}")
    
    print("\nPredicciones realizadas:")
    for result in results:
        print(f"  Muestra {result['sample_id']}: {result['predicted_strength']:.2f} MPa ({result['classification']})")
    
    print(f"\nGráficos guardados en: concrete_prediction_results.png")

if __name__ == "__main__":
    main()
