# Project 1 – Part 2: Prediction and Model Retraining Machine Learning Application

## Context of the project

Organizations in the public and social sectors continuously collect citizen opinions and feedback through participation platforms. These open-ended texts contain valuable insights about issues related to health, education, gender equality, and other critical social topics.

However, analyzing these opinions manually is slow, resource-intensive, and difficult to scale. Because the data is unstructured, automatically identifying the main theme of each comment becomes a significant technical challenge.

This project addresses that challenge by developing a text analytics system capable of:

Processing citizen opinions written in natural language.

Automatically identifying which Sustainable Development Goal (SDG) each opinion relates to, focusing on:

- SDG 3 – Good Health and Well-Being

- SDG 4 – Quality Education

- SDG 5 – Gender Equality

Providing a classification result along with prediction probabilities.

The goal is to transform large volumes of unstructured citizen feedback into structured, actionable information, enabling organizations to better understand societal needs, support decision-making, and prioritize social interventions.

## Description of the project
The goal of this project is to implement a REST API capable of performing predictions and retraining a machine learning model. The model classifies user opinions according to the Sustainable Development Goal (SDG) they belong to.
The solution uses pipelines for data processing and follows a framework-based architecture for building the API..


## REST API Structure
The API is composed of two main endpoints:

### 1. **Prediction Endpoint**
This endpoint receives one or more data instances through the request body in JSON format. The JSON must include all required features.
The endpoint processes the input and returns a list of predictions, keeping the same order as the received data.

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
This endpoint allows retraining the model with new labeled data.
The input must include both the features and the target variable (sdg).

After retraining, the updated model replaces the previous version so future predictions use the newly trained model.
The endpoint also returns performance metrics such as **Precision**, **Recall** and **F1-score**.

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



# Instructions to Run the Application 
Follow these steps to run the prediction and retraining application.
## 1. Create a Python Virtual Environment

It is recommended to use a virtual environment to isolate the project dependencies.
Run the following command:

```bash
python -m venv biML
 ```

This will create a virtual environment named biML.

## 2. Activate the Virtual Environment
Activate it depending on your operating system:
 -  Windows
 ```
.\biML\Scripts\activate
 ```
-  MacOs o Linux
 ```
source biML/bin/activate
 ```

## Install Dependencies

With the environment activated, install the required libraries:

 ```bash
pip install fastapi uvicorn pandas scikit-learn nltk contractions openpyxl
 ```

Main installed dependencies include:

    fastapi (0.115.0)
    uvicorn (0.31.0)
    pandas (2.2.3)
    scikit-learn (1.5.2)
    nltk (3.9.1)
    contractions (0.1.73)
    openpyxl (3.1.5)

## 4.Start the Application 
Run the API using uvicorn:

 ```bash
uvicorn mlAPI:app --reload
 ```

This launches the server in autoreload mode, so the application updates automatically when the code changes.
The API will be available at:

 ```bash
http://127.0.0.1:8000
 ```
