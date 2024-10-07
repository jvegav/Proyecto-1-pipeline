# Proyecto 1 - Parte 2: Aplicación de Predicción y Reentrenamiento de Modelo de Machine Learning

## Descripción del Proyecto
Este proyecto tiene como objetivo la implementación de una **API REST** para la predicción y reentrenamiento de un modelo de machine learning que clasifica opiniones en función del ODS (Objetivo de Desarrollo Sostenible) al que pertenecen. Utiliza **pipelines** para el procesamiento de los datos y un **framework** para el desarrollo de la API.

## Estructura de la API REST
La API está compuesta por dos endpoints principales:

### 1. **Endpoint de Predicción**
Este endpoint recibe una o más instancias de datos a través del cuerpo de la solicitud (body) en formato JSON, conteniendo todas las características requeridas. El endpoint procesa estas instancias y devuelve una lista de predicciones, respetando el orden de los datos recibidos.

- **Método HTTP:** `POST`
- **URL:** `/predict`
- **Request:**
    ```json
    {
      "Textos_espanol": ["Texto de ejemplo 1", "Texto de ejemplo 2"]
    }
    ```
- **Response:**
    ```json
    {
      "prediction": [1, 2]
    }
    ```

### 2. **Endpoint de Reentrenamiento**
Este segundo endpoint permite reentrenar el modelo con nuevas instancias de datos, que incluyen tanto las características como la variable objetivo (`sdg`). Tras el reentrenamiento, el modelo actualizado reemplaza la versión anterior para que las futuras predicciones utilicen este nuevo modelo. Además, el endpoint devuelve métricas de desempeño del modelo, como **Precision**, **Recall** y **F1-score**.

- **Método HTTP:** `POST`
- **URL:** `/train`
- **Request:**
    ```json
    {
      "Textos_espanol": ["Texto de entrenamiento 1", "Texto de entrenamiento 2"],
      "sdg": [1, 2]
    }
    ```
- **Response:**
    ```json
    {
      "precision": 0.85,
      "recall": 0.80,
      "f1_score": 0.82
    }
    ```


