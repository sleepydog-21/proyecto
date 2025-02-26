# Proyecto de Machine Learning

Este repositorio contiene el código, datos y modelos para un proyecto de machine learning. El objetivo del proyecto es ser un Prototipo de un modelo en Python que permita estimar el precio de una casa dadas algunas características que el usuario deberá proporcionar a través de un front al momento de la inferencia.
Datos:.

---

## Estructura del Repositorio
```
.
├── notebooks/
├── data/
│   ├── Raw/
│   └── Processed/
├── model/
├── src/
│   ├── analysis/
│   ├── preprocessing/
│   └── training/
├── scripts/
│   ├── prep.py
│   ├── train.py
│   └── inference.py
└── README.md
```


---

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado Python 3.x y las siguientes librerías:

- pandas
- numpy
- scikit-learn
- joblib

Puedes instalar las dependencias ejecutando:

```bash
pip install -r requirements.txt

##Instrucciones de Uso
#1. Preprocesamiento de Datos
Ejecuta el script prep.py para preprocesar los datos crudos. Los datos procesados se guardarán en data/Processed/.

bash
Copy
python scripts/prep.py
#2. Entrenamiento del Modelo
Ejecuta el script train.py para entrenar el modelo. El modelo entrenado se guardará en model/modelo_optimizado.pkl.

bash
Copy
python scripts/train.py
#3. Inferencia
Ejecuta el script inference.py para realizar predicciones. Las predicciones se guardarán en data/predictions/.

bash
Copy
python scripts/inference.py

##Módulos y Funciones
src/preprocessing/limpieza.py
Contiene funciones para limpiar y preprocesar los datos.

def clean_data(raw_data):
    """
    Limpia los datos crudos.

    Parámetros:
    - raw_data (pd.DataFrame): Datos crudos.

    Retorna:
    - pd.DataFrame: Datos limpios.
    """
    # Lógica de limpieza
    cleaned_data = raw_data.dropna()
    return cleaned_data
