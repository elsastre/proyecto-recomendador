from src.preprocess import DataProcessor
from src.model import create_ncf_model
import os

# 1. Preparar datos
processor = DataProcessor('data/u.data')
df = processor.load_and_clean()
X_train, X_test, y_train, y_test = processor.get_train_test(df)

num_users = df['user_idx'].nunique()
num_movies = df['movie_idx'].nunique()

# 2. Construir Modelo
model = create_ncf_model(num_users, num_movies)

# 3. Entrenar
print("Iniciando entrenamiento...")
history = model.fit(
    [X_train[:, 0], X_train[:, 1]], 
    y_train,
    batch_size=64,
    epochs=10,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    verbose=1
)

# 4. Guardar (Paso clave para el MLE)
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/recommender_v1.h5')
print("Modelo guardado en models/recommender_v1.h5")