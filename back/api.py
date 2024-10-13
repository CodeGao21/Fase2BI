from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
from model_utils import create_pipeline, save_model, load_data
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Configurar logging
logging.basicConfig(filename='logs/api.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Inicializar la aplicación FastAPI
app = FastAPI()

# Configurar CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permitir solicitudes desde estos orígenes
    allow_credentials=True,
    allow_methods=["*"],    # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],    # Permitir todas las cabeceras
)

# Cargar el modelo más reciente al iniciar la aplicación
models_directory = 'models'
model_files = [f for f in os.listdir(models_directory) if f.endswith('.pkl')]
if model_files:
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_directory, x)))
    model_path = os.path.join(models_directory, latest_model)
    model = joblib.load(model_path)
    logging.info(f"Modelo cargado: {latest_model}")
else:
    logging.error("No se encontró ningún modelo entrenado. Asegúrate de entrenar el modelo primero.")
    model = None

# Definir las clases de solicitud
class PredictionRequest(BaseModel):
    texts: list

class RetrainRequest(BaseModel):
    texts: list
    labels: list

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    logging.info("Solicitud de predicción recibida")
    texts = request.texts
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    predicted_probabilities = probabilities.max(axis=1)
    return {
        "predictions": predictions.tolist(),
        "probabilities": predicted_probabilities.tolist()
    }

@app.post("/retrain")
def retrain(request: RetrainRequest):
    global model
    logging.info("Solicitud de reentrenamiento recibida")
    new_texts = request.texts
    new_labels = request.labels
    if len(new_texts) != len(new_labels):
        raise HTTPException(status_code=400, detail="El número de textos y etiquetas debe ser igual")
    
    # Cargar datos antiguos
    data_file = 'data/data.xlsx'
    old_texts, old_labels = load_data(data_file)
    
    # Combinar datos antiguos y nuevos
    texts = list(old_texts) + new_texts
    labels = list(old_labels) + new_labels
    
    # Dividir los datos combinados en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # Crear un nuevo pipeline y reentrenar
    model = create_pipeline()
    model.fit(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de validación
    y_pred = model.predict(X_val)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    # Guardar el modelo actualizado
    save_model(model)
    logging.info("Modelo reentrenado y guardado exitosamente")
    
    # Devolver las métricas de desempeño
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "message": "Modelo reentrenado y guardado exitosamente"
    }
    return metrics
