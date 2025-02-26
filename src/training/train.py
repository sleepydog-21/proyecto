# %% [markdown]
# Análisis

# %% [markdown]
# IMprotamos las paqueterías necesarias

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# %% [markdown]
# E importamos las herramientas a utilizar

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb





# %% [markdown]
# Definimos el path del archivo

# %%
# Define the file path
file_path = '../../data/finales/finales_data.csv'

# %% [markdown]
# Y cargamos el archivo con las variables limpias y solo las que nos intertesan

# %%
# Read the CSV file
data = pd.read_csv(open(file_path))

# Display the first few rows of the dataframe
print(data.head())

# %% [markdown]
# Separamos la variable objetivo SalePrice de las demás

# %%
# Separar la variable objetivo (SalePrice)
X = data.drop(columns=['SalePrice'])  # Variables predictoras
y = data['SalePrice']  # Variable objetivo (SalePrice)

# %% [markdown]
# Dividimos nuestro data set 80% 20% en conjunto de prueba y de entrenamiento

# %%
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Como primero modelo proponemos un GradientBoostingRegressor

# %%


# Inicializar el modelo de Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Raíz cuadrada del error cuadrático medio
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Graficar la comparación entre los valores reales y las predicciones
plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales de SalePrice")
plt.ylabel("Predicciones de SalePrice")
plt.title("Comparación entre valores reales y predicciones")
plt.show()

# %% [markdown]
# Que nos dá un 28835.631824938075, que nos servirá como benchmarck para comparar con los modelos que se entrenarán a continuación.

# %% [markdown]
# Para mejorar el moedolo anterior, se utilizó un GridSearchCV, y después de muchas iteracciones, se encontró que los parámetros de búsqueda     'n_estimators': [ 150, 200, 250, ],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7] nos dan resultados desentes.

# %%
# Definir los parámetros para la búsqueda
param_grid = {
    'n_estimators': [ 150, 200, 250, ],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Inicializar GridSearchCV
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')

# Ajustar el modelo a los datos
grid_search.fit(X_train, y_train)

# Mostrar el mejor modelo encontrado
print(f"Mejores parámetros: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evaluar el modelo optimizado
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = mse_best ** 0.5
print(f"RMSE del modelo optimizado: {rmse_best}")

# %% [markdown]
# Los meores parámetros encontrados son: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 250}.Esto mejora nuestro modelo, dandonos un error medio cuadrado de  24873.62355174363 y

# %%
# Crear el modelo LightGBM
lgb_model = lgb.LGBMRegressor(random_state=42)

# Definir los parámetros para la búsqueda
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'num_leaves': [31, 50, 100],
    'subsample': [0.6, 0.8, 1.0]
}

# Inicializar GridSearchCV
grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Ajustar el modelo a los datos
grid_search.fit(X_train, y_train)

# Mostrar el mejor modelo encontrado
print(f"Mejores parámetros: {grid_search.best_params_}")
best_lgb_model = grid_search.best_estimator_

# Evaluar el modelo optimizado
y_pred_best_lgb = best_lgb_model.predict(X_test)

# Calcular el RMSE sobre los valores logarítmicos
mse_best_lgb = mean_squared_error(np.log(y_test), np.log(y_pred_best_lgb))
rmse_best_lgb = np.sqrt(mse_best_lgb)

print(f"RMSE del modelo LightGBM optimizado: {rmse_best_lgb}")


# %% [markdown]
# Lo cual, para los parámetros de {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 250, 'num_leaves': 31, 'subsample': 0.6}; nos dá un log RMSE del modelo LightGBM optimizado: 0.1392329269246963; el mejor hasta.
# 

# %% [markdown]
# Finalmente, guardamos el modelo en ../modelo/

# %%
# Guardar el modelo
joblib.dump(best_model, '../modelo/modelo_optimizado.pkl')

print("Modelo guardado exitosamente en '../../model/modelo_optimizado.pkl'")

# %%



