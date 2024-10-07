import numpy as np
import pandas as pd
import contractions
from joblib import dump, load

import nltk
nltk.download('all') 
from sklearn.base import BaseEstimator, ClassifierMixin
import re, string, unicodedata
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

class ChangeCharacters(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def replace_common_mismatches(self, text):
        common_mismatches = {
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã‘': 'Ñ',
            'Ã±': 'ñ',
            'Ã€': 'À',
            'Ã‚': 'Â',
            'Ãƒ': 'ƒ',
            'Ãˆ': 'È',
            'Ã‰': 'É',
            'Ã‹': 'Ë',
            'ÃŒ': 'Ì',
            'ÃŒ': 'Î',
            'Ãˆ': 'Ò',
            'Ã“': 'Ó',
            'Ã”': 'Ô',
            'Ã•': 'Õ',
            'Ã™': 'Ù',
            'Ã™': 'Û',
            'Ãž': 'Þ',
            'ÃŸ': 'ÿ',
            'Ã': 'á',
        }
        for old, new in common_mismatches.items():
            text = text.replace(old, new)
        return text

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].apply(self.replace_common_mismatches)
        return X_copy
