import pandas as pd
import numpy as np
import random

def generate_gundam_dataset(n_samples=500):
    np.random.seed(42)
    
    # 1. Definición de Universos y sus años de inicio de "Hype"
    universos = {
        'Universal Century': 1979,
        'G Gundam': 1994,
        'Gundam Wing': 1995,
        'Gundam SEED': 2002,
        'Gundam 00': 2007,
        'Iron-Blooded Orphans': 2015,
        'Witch from Mercury': 2022
    }
    
    # 2. Configuración de Grados y sus rangos de precios/escalas
    grados_config = {
        'SD':  {'escala': 'SD',    'precio_min': 600,   'precio_max': 1500},
        'HG':  {'escala': '1/144', 'precio_min': 1200,  'precio_max': 3500},
        'RG':  {'escala': '1/144', 'precio_min': 2500,  'precio_max': 4500},
        'MG':  {'escala': '1/100', 'precio_min': 4000,  'precio_max': 9500},
        'PG':  {'escala': '1/60',  'precio_min': 12000, 'precio_max': 28000}
    }

    data = []
    current_year = 2024

    for _ in range(n_samples):
        # Selección de Grado
        grado = random.choice(list(grados_config.keys()))
        config = grados_config[grado]
        
        # Selección de Universo y Año coherente
        universo = random.choice(list(universos.keys()))
        anio_inicio = universos[universo]
        anio_lanzamiento = random.randint(anio_inicio, current_year)
        
        # Variable Hype: 1 si se lanzó en los primeros 3 años de la serie
        hype = 1 if (anio_lanzamiento - anio_inicio) <= 3 else 0
        
        # Precio y Escala
        precio = random.randint(config['precio_min'], config['precio_max'])
        escala = config['escala']
        
        # Distribución
        dist = random.choices(['Regular', 'P-Bandai', 'Exclusiva'], weights=[70, 20, 10])[0]
        
        # Lógica de Ventas Totales (Target Builder)
        # Las ventas dependen del Hype, el Grado (HG vende más volumen) y la antigüedad
        base_ventas = {
            'HG': 50000, 'SD': 40000, 'MG': 25000, 'RG': 20000, 'PG': 5000
        }[grado]
        
        # Factor antigüedad: kits viejos tienen más tiempo para acumular ventas
        factor_tiempo = (current_year - anio_lanzamiento + 1) * 0.1
        
        # Ruido aleatorio para realismo
        ruido = np.random.normal(1, 0.2)
        
        # El "éxito" real (lo que queremos predecir)
        # Se ve afectado positivamente por el hype y si es distribución Regular
        multiplicador_exito = (1.5 if hype == 1 else 1.0) * (0.6 if dist == 'P-Bandai' else 1.2)
        
        ventas_totales = int(base_ventas * factor_tiempo * multiplicador_exito * ruido)

        data.append([
            grado, universo, anio_lanzamiento, precio, 
            escala, dist, hype, ventas_totales
        ])

    df = pd.DataFrame(data, columns=[
        'grado', 'serie_universo', 'anio_lanzamiento', 'precio_salida', 
        'escala', 'tipo_distribucion', 'hype', 'ventas_totales'
    ])
    
    # 3. Crear el Target (0 a 1) normalizado por GRADO
    # Es más justo normalizar por grado porque un PG nunca venderá lo que un HG
    df['exito_comercial'] = df.groupby('grado')['ventas_totales'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    
    return df

# # Generar el dataset
# df_gundam = generate_gundam_dataset(800)
# print(df_gundam.head(10))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df=generate_gundam_dataset(900)

# 1. Preparación de datos (X e y)
X = df.drop(['ventas_totales', 'exito_comercial'], axis=1)
y = df['exito_comercial']

# Dividimos antes de transformar para evitar que información de prueba "contamine" el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PASO A: TRATAR CATEGORÍAS (One-Hot Encoding) ---
# Creamos el traductor y lo "ajustamos" SOLO con los datos de entrenamiento
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train[['grado', 'serie_universo', 'escala', 'tipo_distribucion']])

# Transformamos tanto entrenamiento como prueba
cat_train = encoder.transform(X_train[['grado', 'serie_universo', 'escala', 'tipo_distribucion']])
cat_test = encoder.transform(X_test[['grado', 'serie_universo', 'escala', 'tipo_distribucion']])

# Convertimos a DataFrame para volver a unirlo después
cat_cols = encoder.get_feature_names_out()
df_cat_train = pd.DataFrame(cat_train, columns=cat_cols, index=X_train.index)
df_cat_test = pd.DataFrame(cat_test, columns=cat_cols, index=X_test.index)


# --- PASO B: TRATAR NÚMEROS (StandardScaler) ---
scaler = StandardScaler()
scaler.fit(X_train[['anio_lanzamiento', 'precio_salida', 'hype']]) # 'hype' es 0/1, pero se puede escalar

num_train = scaler.transform(X_train[['anio_lanzamiento', 'precio_salida', 'hype']])
num_test = scaler.transform(X_test[['anio_lanzamiento', 'precio_salida', 'hype']])

df_num_train = pd.DataFrame(num_train, columns=['anio', 'precio', 'hype_esc'], index=X_train.index)
df_num_test = pd.DataFrame(num_test, columns=['anio', 'precio', 'hype_esc'], index=X_test.index)


# --- PASO C: UNIR TODO (Concatenación) ---
X_train_final = pd.concat([df_num_train, df_cat_train], axis=1)
X_test_final = pd.concat([df_num_test, df_cat_test], axis=1)


# --- PASO D: ENTRENAMIENTO ---
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train_final, y_train)

# --- PASO E: EVALUACIÓN ---
predicciones = modelo.predict(X_test_final)
print(f"MAE: {mean_absolute_error(y_test, predicciones):.4f}")
print(f"R² Score: {r2_score(y_test, predicciones):.4f}")



"""
ALTERNATIVA PROFESIONAL (Pipeline):
Si quisieras hacer todo lo anterior en 3 líneas:
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(...)),
    ('regressor', RandomForestRegressor())
])
pipeline.fit(X_train, y_train)
"""

def predecir_exito_kit(grado, universo, anio, precio, escala, distribucion, hype):
    """
    Simula la decisión de Bandai para un nuevo lanzamiento.
    """
    # 1. Crear un DataFrame con el nuevo kit (respetando los nombres de columnas originales)
    nuevo_kit = pd.DataFrame([[grado, universo, anio, precio, escala, distribucion, hype]], 
                             columns=['grado', 'serie_universo', 'anio_lanzamiento', 'precio_salida', 'escala', 'tipo_distribucion', 'hype'])

    # 2. Transformación Manual A: Categorías (Usa el encoder que ya existe)
    cat_nuevo = encoder.transform(nuevo_kit[['grado', 'serie_universo', 'escala', 'tipo_distribucion']])
    df_cat_nuevo = pd.DataFrame(cat_nuevo, columns=encoder.get_feature_names_out())

    # 3. Transformación Manual B: Números (Usa el scaler que ya existe)
    num_nuevo = scaler.transform(nuevo_kit[['anio_lanzamiento', 'precio_salida', 'hype']])
    df_num_nuevo = pd.DataFrame(num_nuevo, columns=['anio', 'precio', 'hype_esc'])

    # 4. Unir las partes
    X_nuevo_final = pd.concat([df_num_nuevo, df_cat_nuevo], axis=1)

    # 5. Predicción
    score = modelo.predict(X_nuevo_final)[0]
    
    # 6. Formatear salida
    print(f"--- 📊 Reporte de Simulación Bandai ---")
    print(f"Kit: {grado} de {universo}")
    print(f"Score de Éxito Estimado: {score:.2f}")
    
    if score > 0.75:
        print("Veredicto: 🟢 ¡Éxito Asegurado! Proceder con producción masiva.")
    elif score > 0.45:
        print("Veredicto: 🟡 Rendimiento Moderado. Considerar tirada estándar.")
    else:
        print("Veredicto: 🔴 Riesgo de bajo rendimiento. ¿Quizás hacerlo P-Bandai?")
        
    return score

# --- EJEMPLO DE USO ---
# Imagina que quieres ver qué pasa si lanzas un MG de SEED hoy mismo
mi_score = predecir_exito_kit('MG', 'Gundam SEED', 2024, 5500, '1/100', 'P-Bandai', 1)

