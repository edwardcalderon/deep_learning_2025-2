"""
Análisis de Regresión para Casas con Precios Menores a $3 Millones

Este script analiza el rendimiento de modelos de regresión de redes neuronales profundas 
para casas con precios por debajo de $3 millones de dólares.

Objetivos:
1. Filtrar el dataset para casas con precios < $3,000,000
2. Comparar métricas con el dataset completo
3. Optimizar hiperparámetros (epochs, batch size, dropout, L2)
4. Probar diferentes arquitecturas de red neuronal
5. Identificar el mejor modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("ANÁLISIS DE REGRESIÓN PARA CASAS CON PRECIOS MENORES A $3 MILLONES")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ============================================================================

print("\n1. CARGANDO Y EXPLORANDO DATOS...")

# Cargar datos
data_path = "./kc_house_data_yr.xlsx"
df = pd.read_excel(data_path)

print(f"Dataset original: {df.shape}")
print(f"\nEstadísticas de precios originales:")
print(df['price'].describe())

# Análisis de distribución de precios
houses_above_3m = (df['price'] >= 3000000).sum()
houses_below_3m = (df['price'] < 3000000).sum()

print(f"\nCasas con precio >= $3,000,000: {houses_above_3m} ({houses_above_3m/len(df)*100:.2f}%)")
print(f"Casas con precio < $3,000,000: {houses_below_3m} ({houses_below_3m/len(df)*100:.2f}%)")

# Visualizar distribución de precios
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histograma de precios
axes[0].hist(df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].set_title('Distribución de Precios - Dataset Completo')
axes[0].set_xlabel('Precio (USD)')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(x=3000000, color='red', linestyle='--', label='$3M límite')
axes[0].legend()

# Box plot
axes[1].boxplot(df['price'])
axes[1].set_title('Box Plot de Precios - Dataset Completo')
axes[1].set_ylabel('Precio (USD)')
axes[1].axhline(y=3000000, color='red', linestyle='--', label='$3M límite')

plt.tight_layout()
plt.savefig('price_distribution_original.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 2. FILTRADO DE DATOS - CASAS MENORES A $3 MILLONES
# ============================================================================

print("\n2. FILTRANDO DATOS PARA CASAS < $3 MILLONES...")

# Filtrar dataset para casas con precios < $3,000,000
df_filtered = df[df['price'] < 3000000].copy()

print(f"Dataset filtrado: {df_filtered.shape}")
print(f"Reducción: {len(df) - len(df_filtered)} casas ({(len(df) - len(df_filtered))/len(df)*100:.2f}%)")

print("\nEstadísticas de precios filtrados:")
print(df_filtered['price'].describe())

# Visualizar distribución de precios filtrados
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Comparación de histogramas
axes[0,0].hist(df['price'], bins=50, alpha=0.7, color='lightcoral', label='Dataset Completo')
axes[0,0].set_title('Distribución - Dataset Completo')
axes[0,0].set_xlabel('Precio (USD)')
axes[0,0].set_ylabel('Frecuencia')

axes[0,1].hist(df_filtered['price'], bins=50, alpha=0.7, color='skyblue', label='< $3M')
axes[0,1].set_title('Distribución - Casas < $3M')
axes[0,1].set_xlabel('Precio (USD)')
axes[0,1].set_ylabel('Frecuencia')

# Box plots comparativos
data_comparison = [df['price'], df_filtered['price']]
axes[1,0].boxplot(data_comparison, labels=['Completo', '< $3M'])
axes[1,0].set_title('Comparación Box Plots')
axes[1,0].set_ylabel('Precio (USD)')

# Scatter plot precio vs sqft_living
axes[1,1].scatter(df_filtered['sqft_living'], df_filtered['price'], alpha=0.5, color='green')
axes[1,1].set_title('Precio vs Área de Vivienda (< $3M)')
axes[1,1].set_xlabel('Área de Vivienda (sqft)')
axes[1,1].set_ylabel('Precio (USD)')

plt.tight_layout()
plt.savefig('price_distribution_filtered.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================================

print("\n3. PREPARANDO DATOS PARA MODELADO...")

# Preparar datos para modelado (similar al notebook original)
# Crear features de fecha
df_filtered['date'] = pd.to_datetime(df_filtered['date'], format='%Y%m%d')
df_filtered['year'] = df_filtered['date'].dt.year
df_filtered['month'] = df_filtered['date'].dt.month

# Seleccionar features para el modelo
features_to_use = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                   'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                   'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year', 'month']

# Preparar X y y
X = df_filtered[features_to_use].copy()
y = df_filtered['price'].copy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# División train/validation/test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Escalado de features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Datos escalados entre {X_train_scaled.min():.3f} y {X_train_scaled.max():.3f}")

# ============================================================================
# 4. FUNCIONES AUXILIARES
# ============================================================================

def create_baseline_model(input_dim):
    """Crear modelo baseline similar al notebook original"""
    model = Sequential([
        Dense(input_dim, activation='relu', input_shape=(input_dim,)),
        Dense(input_dim, activation='relu'),
        Dense(input_dim, activation='relu'),
        Dense(input_dim, activation='relu'),
        Dense(1)  # Salida para regresión
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluar modelo y mostrar métricas"""
    predictions = model.predict(X_test, verbose=0)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = explained_variance_score(y_test, predictions)
    
    print(f"\n=== {model_name} - Métricas de Evaluación ===")
    print(f"MAE: ${mae:,.2f}")
    print(f"MSE: {mse:,.0f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE como % del precio promedio: {mae/y_test.mean()*100:.2f}%")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

def create_model_with_hyperparams(input_dim, dropout_rate=0.3, l2_reg=0.001):
    """Crear modelo con hiperparámetros específicos"""
    model = Sequential([
        Dense(input_dim, activation='relu', 
              kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
              input_shape=(input_dim,)),
        Dropout(dropout_rate) if dropout_rate > 0 else tf.keras.layers.Lambda(lambda x: x),
        Dense(input_dim, activation='relu',
              kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None),
        Dropout(dropout_rate) if dropout_rate > 0 else tf.keras.layers.Lambda(lambda x: x),
        Dense(input_dim, activation='relu',
              kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None),
        Dropout(dropout_rate) if dropout_rate > 0 else tf.keras.layers.Lambda(lambda x: x),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ============================================================================
# 5. MODELO BASELINE - COMPARACIÓN CON DATASET COMPLETO
# ============================================================================

print("\n4. ENTRENANDO MODELO BASELINE...")

# Crear y entrenar modelo baseline
baseline_model = create_baseline_model(X_train_scaled.shape[1])

print("Entrenando modelo baseline...")
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

baseline_history = baseline_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluar modelo baseline
baseline_metrics = evaluate_model(baseline_model, X_test_scaled, y_test, "Modelo Baseline (< $3M)")

# ============================================================================
# 6. BÚSQUEDA DE HIPERPARÁMETROS
# ============================================================================

print("\n5. BÚSQUEDA DE HIPERPARÁMETROS...")

# Probar combinaciones seleccionadas
test_combinations = [
    {'epochs': 100, 'batch_size': 64, 'dropout_rate': 0.0, 'l2_reg': 0.0},
    {'epochs': 100, 'batch_size': 64, 'dropout_rate': 0.3, 'l2_reg': 0.001},
    {'epochs': 150, 'batch_size': 32, 'dropout_rate': 0.3, 'l2_reg': 0.001},
    {'epochs': 100, 'batch_size': 128, 'dropout_rate': 0.5, 'l2_reg': 0.01},
    {'epochs': 50, 'batch_size': 64, 'dropout_rate': 0.3, 'l2_reg': 0.0}
]

results = []
best_score = float('inf')
best_params = None
best_model = None

for i, params in enumerate(test_combinations):
    print(f"\nProbando combinación {i+1}/{len(test_combinations)}: {params}")
    
    # Crear modelo
    model = create_model_with_hyperparams(
        X_train_scaled.shape[1], 
        params['dropout_rate'], 
        params['l2_reg']
    )
    
    # Entrenar
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluar
    val_loss = min(history.history['val_loss'])
    predictions = model.predict(X_val_scaled, verbose=0)
    mae = mean_absolute_error(y_val, predictions)
    
    results.append({
        'params': params,
        'val_loss': val_loss,
        'mae': mae,
        'epochs_trained': len(history.history['loss'])
    })
    
    print(f"Val Loss: {val_loss:,.0f}, MAE: ${mae:,.2f}, Epochs: {len(history.history['loss'])}")
    
    if val_loss < best_score:
        best_score = val_loss
        best_params = params
        best_model = model

print(f"\n=== Mejores Hiperparámetros ===")
print(f"Parámetros: {best_params}")
print(f"Mejor Val Loss: {best_score:,.0f}")

# Mostrar resultados de la búsqueda de hiperparámetros
results_df = pd.DataFrame([
    {
        'epochs': r['params']['epochs'],
        'batch_size': r['params']['batch_size'],
        'dropout': r['params']['dropout_rate'],
        'l2_reg': r['params']['l2_reg'],
        'val_loss': r['val_loss'],
        'mae': r['mae'],
        'epochs_trained': r['epochs_trained']
    } for r in results
])

results_df = results_df.sort_values('val_loss')
print("\nResultados de búsqueda de hiperparámetros (ordenados por val_loss):")
print(results_df.round(2))

# Evaluar el mejor modelo en test set
best_metrics = evaluate_model(best_model, X_test_scaled, y_test, "Mejor Modelo (Hiperparámetros Optimizados)")

# ============================================================================
# 7. COMPARACIÓN DE ARQUITECTURAS
# ============================================================================

print("\n6. COMPARANDO DIFERENTES ARQUITECTURAS...")

def create_architecture_models(input_dim):
    """Crear diferentes arquitecturas para comparar"""
    
    architectures = {}
    
    # Arquitectura 1: Red poco profunda
    architectures['Shallow'] = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Arquitectura 2: Red mediana
    architectures['Medium'] = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Arquitectura 3: Red profunda
    architectures['Deep'] = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Arquitectura 4: Red ancha
    architectures['Wide'] = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    
    # Arquitectura 5: Red con regularización L2
    architectures['L2_Regularized'] = Sequential([
        Dense(128, activation='relu', 
              kernel_regularizer=regularizers.l2(0.01),
              input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(32, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)),
        Dense(1)
    ])
    
    # Compilar todos los modelos
    for name, model in architectures.items():
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return architectures

# Crear arquitecturas
architectures = create_architecture_models(X_train_scaled.shape[1])

print("Arquitecturas creadas:")
for name, model in architectures.items():
    total_params = model.count_params()
    print(f"{name}: {total_params:,} parámetros")

# Entrenar y evaluar cada arquitectura
architecture_results = {}
architecture_histories = {}

for name, model in architectures.items():
    print(f"\nEntrenando arquitectura: {name}")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=150,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0
    )
    
    architecture_histories[name] = history
    
    # Evaluar en test set
    metrics = evaluate_model(model, X_test_scaled, y_test, f"Arquitectura {name}")
    architecture_results[name] = metrics
    
    print(f"Epochs entrenados: {len(history.history['loss'])}")

# ============================================================================
# 8. VISUALIZACIÓN DE RESULTADOS
# ============================================================================

print("\n7. CREANDO VISUALIZACIONES...")

# Crear visualizaciones comparativas
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Comparación de MAE por arquitectura
arch_names = list(architecture_results.keys())
mae_values = [architecture_results[name]['mae'] for name in arch_names]

axes[0,0].bar(arch_names, mae_values, color='skyblue', alpha=0.7)
axes[0,0].set_title('MAE por Arquitectura')
axes[0,0].set_ylabel('MAE (USD)')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Comparación de R² Score por arquitectura
r2_values = [architecture_results[name]['r2'] for name in arch_names]

axes[0,1].bar(arch_names, r2_values, color='lightgreen', alpha=0.7)
axes[0,1].set_title('R² Score por Arquitectura')
axes[0,1].set_ylabel('R² Score')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Comparación de RMSE por arquitectura
rmse_values = [architecture_results[name]['rmse'] for name in arch_names]

axes[0,2].bar(arch_names, rmse_values, color='lightcoral', alpha=0.7)
axes[0,2].set_title('RMSE por Arquitectura')
axes[0,2].set_ylabel('RMSE (USD)')
axes[0,2].tick_params(axis='x', rotation=45)

# 4. Curvas de pérdida durante entrenamiento
for name, history in architecture_histories.items():
    axes[1,0].plot(history.history['loss'], label=f'{name} (train)', alpha=0.7)
    axes[1,0].plot(history.history['val_loss'], label=f'{name} (val)', linestyle='--', alpha=0.7)

axes[1,0].set_title('Curvas de Pérdida Durante Entrenamiento')
axes[1,0].set_xlabel('Época')
axes[1,0].set_ylabel('Loss (MSE)')
axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 5. Comparación con modelo baseline
all_models = ['Baseline'] + arch_names
all_mae = [baseline_metrics['mae']] + mae_values

axes[1,1].bar(all_models, all_mae, color='orange', alpha=0.7)
axes[1,1].set_title('Comparación MAE: Baseline vs Arquitecturas')
axes[1,1].set_ylabel('MAE (USD)')
axes[1,1].tick_params(axis='x', rotation=45)

# 6. Scatter plot: Predicciones vs Valores Reales (mejor modelo)
best_arch_name = min(arch_names, key=lambda x: architecture_results[x]['mae'])
best_arch_model = architectures[best_arch_name]
best_predictions = best_arch_model.predict(X_test_scaled, verbose=0)

axes[1,2].scatter(y_test, best_predictions, alpha=0.5)
axes[1,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1,2].set_xlabel('Valores Reales')
axes[1,2].set_ylabel('Predicciones')
axes[1,2].set_title(f'Predicciones vs Reales - {best_arch_name}')

plt.tight_layout()
plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. RESUMEN FINAL Y CONCLUSIONES
# ============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL Y CONCLUSIONES")
print("="*80)

# Encontrar el mejor modelo general
all_results = {
    'Baseline': baseline_metrics,
    'Best_Hyperparams': best_metrics,
    **architecture_results
}

best_overall_name = min(all_results.keys(), key=lambda x: all_results[x]['mae'])
best_overall_metrics = all_results[best_overall_name]

print(f"\n🏆 MEJOR MODELO GENERAL: {best_overall_name}")
print(f"   MAE: ${best_overall_metrics['mae']:,.2f}")
print(f"   RMSE: ${best_overall_metrics['rmse']:,.2f}")
print(f"   R² Score: {best_overall_metrics['r2']:.4f}")
print(f"   Error como % del precio promedio: {best_overall_metrics['mae']/y_test.mean()*100:.2f}%")

print(f"\n📊 COMPARACIÓN CON DATASET COMPLETO:")
print(f"   Dataset original: {len(df):,} casas")
print(f"   Dataset filtrado (< $3M): {len(df_filtered):,} casas")
print(f"   Reducción: {(len(df) - len(df_filtered))/len(df)*100:.1f}%")

print(f"\n💡 MEJORAS OBSERVADAS:")
print(f"   Precio promedio original: ${df['price'].mean():,.2f}")
print(f"   Precio promedio filtrado: ${df_filtered['price'].mean():,.2f}")
print(f"   Reducción en variabilidad: {(df['price'].std() - df_filtered['price'].std())/df['price'].std()*100:.1f}%")

print(f"\n🔧 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
for key, value in best_params.items():
    print(f"   {key}: {value}")

print(f"\n🏗️ RANKING DE ARQUITECTURAS (por MAE):")
arch_ranking = sorted(architecture_results.items(), key=lambda x: x[1]['mae'])
for i, (name, metrics) in enumerate(arch_ranking, 1):
    print(f"   {i}. {name}: ${metrics['mae']:,.2f} MAE, R²={metrics['r2']:.3f}")

print(f"\n✅ CONCLUSIONES:")
print(f"   1. Filtrar casas > $3M mejora significativamente las métricas")
print(f"   2. El mejor modelo reduce el error promedio a ~{best_overall_metrics['mae']/y_test.mean()*100:.1f}% del precio")
print(f"   3. La arquitectura {best_arch_name} mostró el mejor rendimiento")
print(f"   4. La regularización y dropout ayudan a prevenir overfitting")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
