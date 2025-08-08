# ETA Dispatch Simulator 🚕📦

A real-time courier dispatch simulator that predicts trip ETA at order time and intelligently assigns couriers using an ML model trained on NYC taxi data.

> Built for practical machine learning, real-time decision-making, and Streamlit deployment.

---

## 🚀 Live App

Accessible on Streamlit Cloud [LINK](https://eta-dispatch.streamlit.app/)

---

## 🔍 What It Does

- Predicts ETA (in seconds) from pickup to drop-off using historical NYC trip data
- Trained XGBoost regression model based on:
  - Great-circle distance
  - Pickup time (hour, weekday, weekend flag)
  - Historical speed
- Simulates dispatching orders to couriers:
  - Chooses best-fit courier based on ETA
  - Tracks work time, availability, and queue
- Outputs simulation metrics:
  - Average ETA, P50, P90
  - Courier utilization
  - Queue ratio

---

## 📁 Project Structure

```
eta-dispatch/
├── app.py                          # Streamlit interface
├── models/
│   └── xgb_eta_model.json          # Trained model
├── data/
│   └── geo/
│       └── zone_distance_matrix.parquet
├── src/
│   ├── data/
│   │   └── ingest_tlc.py           # Raw TLC loading (optional)
│   ├── features/
│   │   └── make_features.py        # Feature engineering
│   ├── models/
│   │   └── train_model.py          # Model training
│   └── dispatch/
│       └── simulator.py            # Core simulation logic
```

---

## ⚙️ Tech Stack

| Purpose           | Tools Used                                  |
|------------------|----------------------------------------------|
| ML Model         | XGBoost, pandas, scikit-learn                |
| Feature Building | NumPy, datetime, parquet                     |
| Simulation       | Python (OOP, dataclasses), ETA prediction    |
| UI / Deployment  | Streamlit, GitHub                            |
| Hosting (Data)   | Hugging Face Datasets (via HTTPS)            |

---

## 🛠 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Satournine/eta-dispatch.git
cd eta-dispatch

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

---

## 📦 Notes

- The processed dataset (`features_yellow_tripdata_2025-06.parquet`) is **not stored in the repo**.
- It will be loaded from Hugging Face via URL inside `simulator.py`. You must have internet access and working SSL certificates.

---

## 📊 Example Output

```
Assignments:
Order 1 assigned to Courier 2 with total ETA 122.6 sec
Order 2 assigned to Courier 1 with total ETA 110.9 sec
...

Simulation Metrics:
- Average ETA: 120.7 seconds
- P50: 118.3 seconds
- P90: 134.6 seconds
- Couriers Utilization: [0.32, 0.27]
- Queued Orders: 2 (20%)
```
