import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.user_map = {}
        self.movie_map = {}

    def load_and_clean(self):
        # Cargamos los datos (user_id, item_id, rating, timestamp)
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(self.filepath, sep='\t', names=columns)
        
        # Mapeo de IDs a índices (0 a N-1) para las capas de Embedding
        # Esto es vital para evitar errores de índice en la red neuronal
        df['user_idx'] = df['user_id'].astype('category').cat.codes
        df['movie_idx'] = df['item_id'].astype('category').cat.codes
        
        self.user_map = dict(enumerate(df['user_id'].astype('category').cat.categories))
        self.movie_map = dict(enumerate(df['item_id'].astype('category').cat.categories))
        
        return df

    def get_train_test(self, df):
        # Escalamos ratings de 1-5 a 0-1 para que la Sigmoide converja mejor
        # y sea tratado como una probabilidad de "gusto".
        X = df[['user_idx', 'movie_idx']].values
        y = df['rating'].values / 5.0 
        
        return train_test_split(X, y, test_size=0.2, random_state=42)