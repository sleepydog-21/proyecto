# %% [markdown]
# Selección e ingieneria de variables

# %% [markdown]
# Cargamos las paqueterías necesarias

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
from sklearn.model_selection import train_test_split


# %% [markdown]
# Establecemos el path de la base de datos

# %%
# Define the file path
file_path = '../../data/Processed/preprocessed_data.csv'

# %% [markdown]
# Y cargamos la base de datos, y visualizamos rapido de las variables

# %%
# Read the CSV file
data = pd.read_csv(open(file_path))

# Display the first few rows of the dataframe
print(data.head())

# %% [markdown]
# Verificamos la forma del abase de datos despues de la lnmpieza y los encoder

# %%
print(data.shape)  # Devuelve (n_filas, n_columnas)

# %% [markdown]
# Calculamos la matrpiz de correlacciones, y mostramos las correlacciones con la vrariable de interes, Saleprice

# %%
# Calcular correlaciones
correlation_matrix = data.corr(numeric_only=True)  # Excluye variables categóricas
saleprice_corr = correlation_matrix["SalePrice"].sort_values(ascending=False)

# Mostrar las más correlacionadas
print(saleprice_corr)

# %% [markdown]
# Como son muchas, nos fijaremos en aquellas que tengan una correlaccion mayor a 0.30

# %%
high_corr_vars = saleprice_corr[abs(saleprice_corr) > 0.3]

# Mostrar las variables filtradas
print(high_corr_vars)

# %%


# %%
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Matriz de correlación")
plt.show()

# %% [markdown]
# Como podemos ver, siguen haabiendo muchas variables poco relaccionadas. Realizamos un EDA rápido con las variables que tienen correlaccion mayora 50% con SalePrice

# %% [markdown]
# #EDA

# %%
# Filtrar las variables con correlación mayor al 50% con 'SalePrice'
correlated_features = saleprice_corr[saleprice_corr.abs() > 0.5].index

# Mostrar las variables correlacionadas con 'SalePrice'
print(correlated_features)

# Visualizar la matriz de correlación entre 'SalePrice' y las variables correlacionadas
plt.figure(figsize=(10, 6))
sns.heatmap(data[correlated_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación entre SalePrice y variables correlacionadas (>50%)')
plt.show()

# Graficar las relaciones entre SalePrice y las variables correlacionadas
for feature in correlated_features:
    if feature != 'SalePrice':  # Excluir 'SalePrice' de la gráfica de dispersión
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data[feature], y=data['SalePrice'])
        plt.title(f'Relación entre SalePrice y {feature}')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()

# %%
# Separar la variable objetivo (SalePrice)
X = data.drop(columns=['SalePrice'])  # Variables predictoras
y = data['SalePrice']  # Variable objetivo (SalePrice)


# %% [markdown]
# Para hacer la base de datos un poco mas ligera, estandarizaremos, y eliminaremos las variables con  una correlaccion menor a 1% con Saleforce, asi como variables altamente correlaccionadas entre si.

# %%
# 1. Estandarizar las variables predictoras
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  
X = pd.DataFrame(X_scaled, columns=X.columns)  # Convertir de nuevo a DataFrame


# %%
# 2. Eliminar variables con baja correlación con SalePrice
correlation = X.join(y).corr(numeric_only=True)["SalePrice"].drop("SalePrice")

threshold = 0.01  # Ajusta según sea necesario
selected_features = correlation[abs(correlation) > threshold].index

X = X[selected_features]  # Mantener solo las columnas relevantes

print(f"Columnas eliminadas por baja correlación con SalePrice: {list(X.columns.difference(selected_features))}")
print(f"Nuevas dimensiones de X después de eliminar baja correlación: {X.shape}")


# %%
# 3. Calcular la matriz de correlación de las variables predictoras
corr_matrix = X.corr().abs()

# 4. Crear un triángulo superior de la matriz de correlación
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 5. Encontrar las columnas altamente correlacionadas (> 0.9)
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

# 6. Eliminar esas columnas
X = X.drop(columns=to_drop)

print(f"Dimensiones del conjunto de datos después de eliminar variables correlacionadas: {X.shape}")

# %%
# Calcular correlaciones
correlation_matrix = X.corr(numeric_only=True)  # Excluye variables categóricas
saleprice_corr = X.join(y).corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

# Mostrar las más correlacionadas
print(saleprice_corr)

# %% [markdown]
# Volvemos a juntar Saleprice con nuestro base de adatos

# %%
X['SalePrice'] = y

# %% [markdown]
# Finalemnte Guardamos nuestra base de datos en ../data/finales/

# %%
# Guardar el dataframe preprocesado en un archivo CSV
X.to_csv('../../data/finales/finales_data.csv', index=False)


