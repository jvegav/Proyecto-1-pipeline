from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from changeCharacters import ChangeCharacters
from textProccessing import ApplyTextProcessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from typing import List
from joblib import dump, load

app = FastAPI()


class PredictingItem(BaseModel):
    Textos_espanol: List[str]

class TrainingItem(BaseModel):
    Textos_espanol: List[str]
    sdg : List[int]


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


data = pd.read_excel('./ODScat_345.xlsx')
data.to_csv('./ODScat_345.csv', index=False)
data = pd.read_csv('./ODScat_345.csv', encoding='utf-8')

X = data.drop(['sdg'], axis=1)
Y = data['sdg']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

pipeline.fit(X_train, Y_train)

dump(pipeline, 'pipeline_model.joblib')


@app.post('/predict')
async def ods_scoring_endpoint(item: PredictingItem):
    pipeline_ods = load('pipeline_model.joblib')
    df = pd.DataFrame(item.Textos_espanol, columns=['Textos_espanol'])
    yhat = pipeline_ods.predict(df)
    return {"prediction": yhat.tolist()}


@app.post('/train')
async def train_model_endpoint(item: TrainingItem):
    df = pd.DataFrame({
        'Textos_espanol': item.Textos_espanol,
        'sdg': item.sdg
    })
    X = df[['Textos_espanol']]
    Y = df['sdg']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    pipeline_ods = load('pipeline_model.joblib')
    pipeline_ods.fit(X_train, Y_train)
    
    y_pred = pipeline_ods.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred, average='weighted')
    f1 = f1_score(Y_test, y_pred, average='weighted')

    dump(pipeline_ods, 'pipeline_model.joblib')

    
    return {
        "message": "Modelo reentrenado exitosamente",
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1
    }

    dump(pipeline_ods, 'pipeline_model.joblib')

    return {"message": "Modelo reentrenado exitosamente"}
