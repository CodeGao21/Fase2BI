import pandas as pd
import nltk
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime
import logging
from sklearn.calibration import CalibratedClassifierCV

# Configurar logging
logging.basicConfig(filename='logs/model.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Descargar stopwords y modelo de SpaCy si es necesario
nltk.download('stopwords')
try:
    nlp = spacy.load('es_core_news_sm')
except:
    spacy.cli.download('es_core_news_sm')
    nlp = spacy.load('es_core_news_sm')

spanish_stopwords = set(stopwords.words('spanish'))

def load_data(file_path):
    logging.info(f"Cargando datos desde {file_path}")
    df = pd.read_excel(file_path)
    X = df['Textos_espanol']
    y = df['sdg']
    return X, y

def lemmatize_and_remove_stopwords(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha]
    tokens_without_stopwords = [token for token in lemmatized_tokens if token.lower() not in spanish_stopwords]
    return ' '.join(tokens_without_stopwords)

def split_data(X, y):
    logging.info("Dividiendo datos en conjuntos de entrenamiento y prueba")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_pipeline():
    logging.info("Creando el pipeline de procesamiento y clasificación")
    pipeSVC = Pipeline([
        ('tfi', TfidfVectorizer(tokenizer=lemmatize_and_remove_stopwords, ngram_range=(1, 4))),
        ('clf', CalibratedClassifierCV(LinearSVC()))
    ])
    return pipeSVC

def train_model(pipeline, X_train, y_train):
    logging.info("Iniciando entrenamiento del modelo")
    pipeline.fit(X_train, y_train)
    logging.info("Entrenamiento completado")
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    logging.info("Evaluando el modelo")
    predictions = pipeline.predict(X_test)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, predictions))
    print("Exactitud:", accuracy_score(y_test, predictions))
    return predictions

def save_model(model, directory='models'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{directory}/text_classification_pipeline_{timestamp}.pkl"
    joblib.dump(model, filename)
    logging.info(f"Modelo guardado en {filename}")
    print(f"Modelo guardado en {filename}")

def load_model(filename):
    logging.info(f"Cargando modelo desde {filename}")
    return joblib.load(filename)
