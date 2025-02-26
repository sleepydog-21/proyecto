# %% [markdown]
# Descarga Y limpieza de los datos

# %% [markdown]
# Importamos las Paqueterías necesarias

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# Definimos el path del archivo

# %%
# Define the file path
file_path = '../../data/raw/train.csv'


# %% [markdown]
# Leemos el archivo, y nos damos una brebe idea de los datos

# %%
# Read the CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head())

# %% [markdown]
# Buscamos columnas con valores faltantes

# %%
print(data.isnull().sum()[data.isnull().sum() > 0])

# %% [markdown]
# Siguiendo la descripcion de los datos, hay variables cuyos valores faltantes son 0
# :LotFrontage, MasVnrArea, GarageYrBlt, mas no requieren otro tipo de ingienería de variables. Por lo que definiremos una funccion que rellee valores faltantes con 0.

# %%
def reemplazar_nan_por_cero(dataframe, columna):
    dataframe[columna] = dataframe[columna].fillna(0)

# %% [markdown]
# Y se los aplicaremos a las variables adecuadas

# %%
# Ejecutar la función para las columnas mencionadas
columnas_a_reemplazar = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for columna in columnas_a_reemplazar:
    reemplazar_nan_por_cero(data, columna)

# %% [markdown]
# Realizamos un mapeo de las variables ordinales a numéricas:

# %%
# Mapeo para cada variable categórica

# Alley
mapeo_alley = {np.nan: 0, 'Grvl': 1, 'Pave': 2}
data['Alley'] = data['Alley'].map(mapeo_alley)

# MasVnrType
mapeo_masvnr = {np.nan: 0, 'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3, 'None': 4}
data['MasVnrType'] = data['MasVnrType'].map(mapeo_masvnr)

# BsmtQual
mapeo_bsmtqual = {np.nan: 0, 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
data['BsmtQual'] = data['BsmtQual'].map(mapeo_bsmtqual)

# BsmtCond
mapeo_bsmtcond = {np.nan: 0, 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
data['BsmtCond'] = data['BsmtCond'].map(mapeo_bsmtcond)

# BsmtExposure
mapeo_bsmtexposure = {np.nan: 0, 'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1}
data['BsmtExposure'] = data['BsmtExposure'].map(mapeo_bsmtexposure)

# BsmtFinType1
mapeo_bsmttype1 = {np.nan: 0, 'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
data['BsmtFinType1'] = data['BsmtFinType1'].map(mapeo_bsmttype1)

# BsmtFinType2
mapeo_bsmttype2 = {np.nan: 0, 'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
data['BsmtFinType2'] = data['BsmtFinType2'].map(mapeo_bsmttype2)

# Electrical
mapeo_electrical = {np.nan: 0, 'SBrkr': 1, 'FuseA': 2, 'FuseF': 3, 'FuseP': 4, 'Mix': 5}
data['Electrical'] = data['Electrical'].map(mapeo_electrical)

# FireplaceQu
mapeo_fireplace = {np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
data['FireplaceQu'] = data['FireplaceQu'].map(mapeo_fireplace)

# GarageType
mapeo_garagetype = {np.nan: 0, 'Attchd': 1, 'Detchd': 2, 'BuiltIn': 3, 'CarPort': 4, 'Basment': 5}
data['GarageType'] = data['GarageType'].map(mapeo_garagetype)

# GarageFinish
mapeo_garagefinish = {np.nan: 0, 'Fin': 1, 'RFn': 2, 'Unf': 3}
data['GarageFinish'] = data['GarageFinish'].map(mapeo_garagefinish)

# GarageQual
mapeo_garagequal = {np.nan: 0, 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
data['GarageQual'] = data['GarageQual'].map(mapeo_garagequal)

# GarageCond
mapeo_garagecond = {np.nan: 0, 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
data['GarageCond'] = data['GarageCond'].map(mapeo_garagecond)

# PoolQC
mapeo_poolqc = {np.nan: 0, 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
data['PoolQC'] = data['PoolQC'].map(mapeo_poolqc)

# Fence
mapeo_fence = {np.nan: 0, 'GdPrv': 1, 'MnPrv': 2, 'GdWo': 3, 'MnWo': 4, 'MnWw': 5}
data['Fence'] = data['Fence'].map(mapeo_fence)

# MiscFeature
mapeo_misc_feature = {np.nan: 0, 'Elev': 1, 'Gar2': 1, 'Othr': 1, 'Shed': 1, 'TenC': 1, 'NA': 0}
data['MiscFeature'] = data['MiscFeature'].map(mapeo_misc_feature)

# ExterQual
mapeo_exterqual = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
data['ExterQual'] = data['ExterQual'].map(mapeo_exterqual)

# ExterCond
mapeo_extercond = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
data['ExterCond'] = data['ExterCond'].map(mapeo_extercond)

# HeatingQC
mapeo_heatingqc = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
data['HeatingQC'] = data['HeatingQC'].map(mapeo_heatingqc)

# KitchenQual
mapeo_kitchenqual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
data['KitchenQual'] = data['KitchenQual'].map(mapeo_kitchenqual)

# CentralAir
mapeo_centralair = {'Y': 1, 'N': 0}
data['CentralAir'] = data['CentralAir'].map(mapeo_centralair)

# PavedDrive
mapeo_paveddrive = {'Y': 1, 'N': 0, 'P': 0}  # Asumiendo que 'P' es una variante de 'N' que representa pavimento parcial.
data['PavedDrive'] = data['PavedDrive'].map(mapeo_paveddrive)

# %% [markdown]
# Garagetype dio roblemas, asi que se realizo de maner diferente

# %%
# Reemplazar los NaN previos con un valor, por ejemplo, 0
data['GarageType'] = data['GarageType'].fillna(np.nan)

# Mapeo de los valores de GarageType
mapeo_garage_type = {
    np.nan: 0,         # No garage (se convierte en 0)
    'Detchd': 2,       # Detached from home
    'BuiltIn': 3,      # Built-In (Garage part of house)
    'CarPort': 4,      # Car Port
    'Basment': 5,      # Basement Garage
    'Attchd': 6,       # Attached to home
    '2Types': 7        # More than one type of garage
}

# Aplicar el mapeo
data['GarageType'] = data['GarageType'].map(mapeo_garage_type).fillna(0)


# %% [markdown]
# verificamos si aun quedan variables categóricas, y reaizamos un one hot encoding

# %%
non_numeric_columns = data.select_dtypes(exclude=['number']).columns
print(non_numeric_columns)

# %%
# Seleccionar las variables no numéricas
non_numeric_columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                       'Exterior2nd', 'Foundation', 'Functional', 'SaleType', 'SaleCondition']

# Aplicar One-Hot Encoding
data = pd.get_dummies(data, columns=non_numeric_columns, drop_first=True)


# %% [markdown]
# Finalmente, por si aún tenemos algun dato faltante, lo remplazamos con a media de la categoría.

# %%
def reemplazar_nan_por_moda(dataframe):
    for columna in dataframe.columns:
        if dataframe[columna].isnull().sum() > 0:
            moda = dataframe[columna].mode()[0]  # Obtener la moda
            dataframe[columna] = dataframe[columna].fillna(moda)  # Reemplazar NaN por la moda
    return dataframe

# Aplicar la función a tu dataframe
data = reemplazar_nan_por_moda(data)


# %% [markdown]
# Verificamos si queda alguna variable tipo objeto

# %%
# Verificar las columnas de tipo 'object' (texto) en el DataFrame
object_columns = data.select_dtypes(include=['object']).columns

# Mostrar las columnas de tipo 'object'
print(object_columns)

# %% [markdown]
# Y le aplicamos "a la fuerza" un Label Encoder a la variable Heating, que resultó la única que quedaba como objeto.

# %%
# Usar LabelEncoder para la columna 'Heating'
label_encoder = LabelEncoder()
data['Heating'] = label_encoder.fit_transform(data['Heating'])

# Verificar si hay valores NaN y reemplazarlos si es necesario
data = data.fillna(data.mode().iloc[0])


# %% [markdown]
# Verificamos si queda alguna variable tipo objeto

# %%
# Verificar las columnas de tipo 'object' (texto) en el DataFrame
object_columns = data.select_dtypes(include=['object']).columns

# Mostrar las columnas de tipo 'object'
print(object_columns)

# %% [markdown]
# y listo, guardamos el set limpio de datos

# %%
# Guardar el dataframe preprocesado en un archivo CSV, en la carpeta ../data/processed  
data.to_csv('../../data/processed/preprocessed_data.csv', index=False)



