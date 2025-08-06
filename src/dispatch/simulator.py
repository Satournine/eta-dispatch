from dataclasses import dataclass
from typing import List
import random
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


model_path = Path("models/xgb_eta_model.json")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model_path)
distance_df = pd.read_parquet("data/geo/zone_distance_matrix.parquet")
speed_df = pd.read_parquet("data/processed/features_yellow_tripdata_2025-06.parquet")


@dataclass
class Order:
    order_id: int
    timestamp: float
    pickup_zone: int
    dropoff_zone: int

@dataclass
class Courier:
    courier_id: int
    current_zone: int
    available_at: float

class DispatchSimulator:
    def __init__(self, couriers: List[Courier], orders: List[Order], eta_predictor):
        self.couriers = couriers
        self.orders = orders
        self.eta_predictor = eta_predictor
        self.assignments = []
    
    def run(self):
        for order in self.orders:
            available = [c for c in self.couriers if c.available_at <= order.timestamp]
            if not available:
                continue
                #need queue stuff here
            
            best_eta = float("inf")
            best_courier = None

            for courier in available:
                to_pickup_eta = self.eta_predictor(order, order.timestamp, courier.current_zone)
                trip_eta = self.eta_predictor(order, order.timestamp, order.pickup_zone)
                total_eta = to_pickup_eta + trip_eta

                if total_eta < best_eta:
                    best_eta = total_eta
                    best_courier = courier
            

            if best_courier:
                best_courier.available_at = order.timestamp + best_eta
                best_courier.current_zone = order.dropoff_zone
                self.assignments.append((order.order_id, best_courier.courier_id, best_eta))


distance_lookup = distance_df.set_index(["PULocationID", "DOLocationID"])["great_circle_km"].to_dict()

# We aggregate the speed dataframe by zone and hour to get historical_speed_kmh
speed_lookup = (
    speed_df.groupby(["PULocationID", "DOLocationID", "pickup_hour"])["historical_speed_kmh"]
    .median()
    .to_dict()
)

def predict_eta(order, current_time, courier_zone):
    pu = order.pickup_zone
    do = order.dropoff_zone

    # Assume simulation starts at 2025-06-01
    dt = datetime(2025, 6, 1) + timedelta(seconds=current_time)
    hour = dt.hour
    weekday = dt.weekday()
    is_weekend = weekday >= 5
    month = dt.month

    distance_km = distance_lookup.get((pu, do), 5.0)  
    historical_speed = speed_lookup.get((pu, do, hour), 20.0)

    features = pd.DataFrame([{
        "great_circle_km": distance_km,
        "pickup_hour": hour,
        "pickup_weekday": weekday,
        "is_weekend": is_weekend,
        "pickup_month": month,
        "historical_speed_kmh": historical_speed,
    }])
    
    log_eta_pred = xgb_model.predict(features)[0]
    eta = np.expm1(log_eta_pred)
    return eta

def initialize_couriers(n: int, zone_ids: List[int]) -> List[Courier]:
    couriers = []
    for i in range(n):
        starting_zone = random.choice(zone_ids)
        couriers.append(Courier(courier_id=i, current_zone=starting_zone, available_at=0.0))
    return couriers

def generate_fake_orders(n:int, zone_ids: List[int], start_time=0.0, interval=60.0) -> List[Order]:
    orders = []
    time = start_time
    for i in range(n):
        pu = random.choice(zone_ids)
        do = random.choice(zone_ids)
        while do == pu:
            do = random.choice(zone_ids)
        orders.append(Order(order_id=i, timestamp=time, pickup_zone=pu, dropoff_zone=do))
        time += interval
    return orders

if __name__ == "__main__":
    zone_ids = list(range(1, 264))
    couriers = initialize_couriers(n=5, zone_ids=zone_ids)
    orders = generate_fake_orders(n=10, zone_ids=zone_ids, interval=30.0)

    sim = DispatchSimulator(couriers, orders, predict_eta)
    sim.run()

    print("\nAssignments:")
    for a in sim.assignments:
        print(f"Order {a[0]} assigned to Courier {a[1]} with total ETA {a[2]:.1f} sec")