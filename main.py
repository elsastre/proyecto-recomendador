import logging
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# --- CONFIGURACIÓN DE LOGGING ---
# Esto permite rastrear qué sucede en el servidor Docker sin usar prints
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("recommender-api")

# --- MODELOS DE DATOS (PYDANTIC) ---
# Definimos el "contrato" de la API: qué datos entran y qué datos salen
class Recommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    confidence_score: float = Field(..., ge=0, le=1) # Validamos que el score sea entre 0 y 1

class PredictResponse(BaseModel):
    user_id: int
    recommendations: List[Recommendation]

# --- INICIALIZACIÓN DE LA APP ---
app = FastAPI(
    title="Neural Collaborative Filtering API",
    description="Servicio de recomendaciones de películas basado en Deep Learning y arquitectura NCF.",
    version="1.1.0"
)

# --- CARGA DE ACTIVOS (Carga perezosa o al inicio) ---
# Nota: En producción, esto podría manejarse con un ciclo de vida de la app (lifespan)
try:
    logger.info("Cargando modelo y datasets...")
    # Cargamos el modelo sin compilar (solo inferencia)
    model = tf.keras.models.load_model('models/recommender_v1.h5', compile=False)
    
    # Cargamos datos para mapeos
    from src.preprocess import DataProcessor
    processor = DataProcessor('data/u.data')
    df = processor.load_and_clean()

    # Cargamos títulos de películas para legibilidad
    cols = ['movie_id', 'title'] + [f'extra_{i}' for i in range(22)]
    items = pd.read_csv('data/u.item', sep='|', names=cols, encoding='latin-1')
    movie_titles = dict(zip(items['movie_id'], items['title']))
    logger.info("Sistema listo para recibir peticiones.")
except Exception as e:
    logger.error(f"Error crítico durante el inicio: {e}")
    raise e

# --- ENDPOINTS ---

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "online", "model_version": "1.1.0"}

@app.get(
    "/recommend/{user_id}", 
    response_model=PredictResponse, 
    tags=["Predictions"]
)
def get_recommendations(
    user_id: int, 
    k: int = Query(5, gt=0, le=50, description="Cantidad de recomendaciones")
):
    """
    Genera recomendaciones personalizadas para un usuario específico.
    """
    logger.info(f"Petición recibida: User ID {user_id}, K={k}")

    # 1. Validar existencia del usuario
    if user_id not in df['user_id'].unique():
        logger.warning(f"Intento de acceso con User ID {user_id} no encontrado.")
        raise HTTPException(status_code=404, detail="Usuario no encontrado en el dataset.")

    try:
        # 2. Preparar candidatos (películas que el usuario NO ha visto)
        user_idx = df[df['user_id'] == user_id]['user_idx'].iloc[0]
        watched_movies = df[df['user_id'] == user_id]['movie_idx'].unique()
        all_movie_indices = df['movie_idx'].unique()
        candidate_movies = np.array([m for m in all_movie_indices if m not in watched_movies])
        
        # 3. Inferencia (Predicción masiva)
        user_input = np.array([user_idx] * len(candidate_movies))
        predictions = model.predict([user_input, candidate_movies], verbose=0).flatten()
        
        # 4. Procesar el Top K
        top_indices = predictions.argsort()[-k:][::-1]
        recommended_movie_indices = candidate_movies[top_indices]
        
        results = []
        for i, idx in enumerate(recommended_movie_indices):
            m_id = processor.movie_map[idx]
            results.append(
                Recommendation(
                    rank=i + 1,
                    movie_id=int(m_id),
                    title=movie_titles.get(m_id, "Unknown Title"),
                    confidence_score=float(predictions[top_indices[i]])
                )
            )
        
        logger.info(f"Éxito: Se generaron {len(results)} recomendaciones para el usuario {user_id}.")
        return PredictResponse(user_id=user_id, recommendations=results)

    except Exception as e:
        logger.error(f"Error durante el proceso de recomendación: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar las recomendaciones.")