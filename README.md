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



# Instrucciones para Ejecutar la Aplicación

Sigue estos pasos para ejecutar la aplicación de predicción y reentrenamiento del modelo de Machine Learning.

## 1. Crear un Entorno Virtual en Python

Es recomendable crear un entorno virtual para aislar las dependencias del proyecto. Para crear un entorno virtual en Python, ejecuta el siguiente comando en tu terminal:

```bash
python -m venv biML
 ```

Esto creará un entorno virtual llamado biML.

## 2. Activar el Entorno Virtual
Dependiendo de tu sistema operativo, activa el entorno virtual:
 - En Windows
 ```
.\biML\Scripts\activate
 ```
- En MacOs o Linux
 ```
source biML/bin/activate
 ```

## 3. Instalar las Dependencias

Con el entorno virtual activado, instala las dependencias necesarias ejecutando el siguiente comando:

instala manualmente las siguientes librerías con:

 ```bash
pip install fastapi uvicorn pandas scikit-learn nltk contractions openpyxl
 ```

Dependiendo de tu configuración, aquí está una lista de las principales dependencias que se instalarán:

    fastapi (0.115.0)
    uvicorn (0.31.0)
    pandas (2.2.3)
    scikit-learn (1.5.2)
    nltk (3.9.1)
    contractions (0.1.73)
    openpyxl (3.1.5)

## 4. Inicializa la Aplicacion

Para ejecutar la aplicación, utiliza uvicorn. Ejecuta el siguiente comando:

 ```bash
uvicorn mlAPI:app --reload
 ```

Esto iniciará el servidor en modo recarga automática, lo que significa que se actualizará automáticamente cuando realices cambios en el código.

La aplicación estará disponible en:

 ```bash
http://127.0.0.1:8000
 ```
