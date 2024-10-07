from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from changeCharacters import ChangeCharacters
from textProccessing import ApplyTextProcessing
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from typing import List

app = FastAPI()

# Clase para recibir una lista de textos
class PredictingItem(BaseModel):
    Textos_espanol: List[str]

# Definición del pipeline con los pasos adecuados
pipeline = Pipeline(steps=[
    ('change_characters', ChangeCharacters(columns=['Textos_espanol'])),
    ('transformations', ColumnTransformer(
        transformers=[
            ('tfidf_texto', Pipeline([
                ('apply_processing', ApplyTextProcessing(text_column='Textos_espanol')),
                ('tfidf', TfidfVectorizer())
            ]), 'Textos_espanol')
        ], remainder='passthrough')),
    ('classifier', KNeighborsClassifier(n_neighbors=100))
])

# Cargar datos y ajustar el pipeline
data = pd.read_excel('./ODScat_345.xlsx')
data.to_csv('./ODScat_345.csv', index=False)
data = pd.read_csv('./ODScat_345.csv', encoding='utf-8')

X = data.drop(['sdg'], axis=1)
Y = data['sdg']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

pipeline.fit(X_train, Y_train)

# Endpoint para recibir múltiples textos
@app.post('/')
async def scoring_endpoint(item: PredictingItem):
    # Crear un DataFrame con todos los textos de la lista
    df = pd.DataFrame(item.Textos_espanol, columns=['Textos_espanol'])
    
    # Predecir los resultados usando el pipeline
    yhat = pipeline.predict(df)
    
    # Devolver las predicciones como lista
    return {"prediction": yhat.tolist()}
