import tensorflow as tf
from tensorflow.keras import layers, Model

def create_ncf_model(num_users, num_movies, embedding_size=50):
    # Entradas de los índices
    user_input = layers.Input(shape=(1,), name='user_input')
    movie_input = layers.Input(shape=(1,), name='movie_input')

    # Capas de Embedding: Transforman un ID en un vector de características latentes
    user_embedding = layers.Embedding(num_users, embedding_size, name='user_emb')(user_input)
    movie_embedding = layers.Embedding(num_movies, embedding_size, name='movie_emb')(movie_input)

    # Aplanamos los vectores
    user_vec = layers.Flatten()(user_embedding)
    movie_vec = layers.Flatten()(movie_embedding)

    # Concatenación: Aquí es donde la "magia" de la red neuronal ocurre
    concat = layers.Concatenate()([user_vec, movie_vec])

    # Capas Densas (MLP): Aprenden interacciones no lineales entre gustos y películas
    dense_1 = layers.Dense(64, activation='relu')(concat)
    dropout_1 = layers.Dropout(0.2)(dense_1)
    dense_2 = layers.Dense(32, activation='relu')(dropout_1)
    
    # Salida: Sigmoide para predecir la probabilidad de rating alto
    output = layers.Dense(1, activation='sigmoid')(dense_2)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model