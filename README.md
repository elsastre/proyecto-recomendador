![Python CI](https://github.com/elsastre/recommender-system/actions/workflows/python-app.yml/badge.svg)

# 🎬 Neural Collaborative Filtering (NCF) Movie Recommender

A production-ready recommendation system based on **Deep Learning**. This project implements an NCF architecture to predict user preferences and serves them through a scalable **FastAPI** service containerized with **Docker**.

## 🚀 Key Features
- **Deep Learning Architecture**: Built with TensorFlow/Keras using Embedding layers and Multi-Layer Perceptron (MLP).
- **Production-Grade API**: Robust backend with data validation (Pydantic) and structured logging.
- **Containerized Environment**: Fully dockerized for consistent deployment across any system.
- **Interactive UI**: User-friendly dashboard built with Streamlit for real-time recommendations.

## 🧠 Architecture
The system uses **Neural Collaborative Filtering**. Instead of simple matrix factorization, it uses a neural network to learn the non-linear interaction between users and items.

$$y_{ui} = \sigma(MLP(P^T v_u \oplus Q^T v_i))$$

![NCF Architecture](./docs/ncf-architecture.png)

- **Input**: User IDs and Movie IDs.
- **Latent Space**: High-dimensional Embeddings.
- **Interaction Layer**: Concatenated vectors passed through Dense layers with ReLU activation.
- **Output**: A probability score (0-1) representing the likelihood of interest.



## 🛠️ Tech Stack
- **Engine**: Python 3.12, TensorFlow, Pandas, NumPy.
- **API**: FastAPI, Uvicorn, Pydantic.
- **DevOps**: Docker.
- **Frontend**: Streamlit.

## 📦 Getting Started

### Prerequisites
- Docker installed on your machine.
- MovieLens 100k dataset (`u.data`, `u.item`) placed in the `/data` folder.

### 1. Train the model (creates the required model file)
Before building the Docker image, you need to train the Neural Collaborative Filtering model.  
Run the training script from your project root:

```bash
python train.py
```

This will:
- Read the dataset from `data/`
- Train the model and save it as `models/recommender_v1.keras`

### 2. Build the Docker image
```bash
docker build -t movie-recommender-api .
```

### 3. Run the container (mounting both data and models)
```bash
docker run --name recommender-service -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  movie-recommender-api
```

### 4. Access API documentation
Open [http://localhost:8000/docs](http://localhost:8000/docs) to see the interactive Swagger UI.

## Running the UI

- **In a separate terminal (with your venv active)**:

    ```bash
    streamlit run src/app_ui.py 
    ```

## 📊 Model Performance

The model was trained for 10 epochs. The best generalization was observed around **Epoch 4**, before the onset of overfitting.

| Metric | Value |
| :--- | :--- |
| Training Loss (MSE) | 0.0480 |
| Validation Loss (MSE) | 0.0540 |
| Validation MAE | 0.1829 |

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

*Note: Ratings are normalized to a [0, 1] scale. An MAE of 0.18 on a 5-star scale represents an average error of approximately 0.9 stars.*

Developed by Braihans - AI Engineering Student




