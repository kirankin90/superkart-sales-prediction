# SuperKart Sales Forecasting - Model Deployment

Predicts `Product_Store_Sales_Total` for a product in a SuperKart store.

* `backend/`  - Flask REST API (`/v1/predict`, `/v1/predictbatch`) serving the serialized scikit-learn / XGBoost pipeline
* `frontend/` - Streamlit UI that calls the backend for single and batch predictions

## Run both containers (GitHub Codespace / any Docker host)

```bash
docker network create superkart-network

cd backend
docker build -t superkart-backend .
docker run -d --name backend --network superkart-network -p 7860:7860 superkart-backend

cd ../frontend
docker build -t superkart-frontend .
docker run -d --name frontend --network superkart-network -p 8501:8501 superkart-frontend
```

* Backend API: port **7860** (make the port Public in Codespaces to call it from outside)
* Streamlit UI: port **8501** (reaches the API at `http://backend:7860` through the Docker network)
