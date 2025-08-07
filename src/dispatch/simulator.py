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
    total_work_time: float = 0.0

class DispatchSimulator:
    def __init__(self, couriers: List[Courier], orders: List[Order], eta_predictor):
        self.couriers = couriers
        self.orders = orders
        self.eta_predictor = eta_predictor
        self.assignments = []
        self.eta_log = []
        self.queued_orders = []

    def run(self):
        for order in self.orders:
            available = [c for c in self.couriers if c.available_at <= order.timestamp]
            if not available:
                self.queued_orders.append(order)
                continue

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
                best_courier.total_work_time += best_eta
                best_courier.available_at = order.timestamp + best_eta
                best_courier.current_zone = order.dropoff_zone
                self.assignments.append((order.order_id, best_courier.courier_id, best_eta))
                self.eta_log.append(best_eta)

    def report_metrics(self):
        if not self.assignments:
            print("No orders were assigned.")
            return

        etas = np.array(self.eta_log)
        end_times = [c.available_at for c in self.couriers if c.available_at > 0]
        total_sim_time = max(end_times) if end_times else 1.0
        avg_eta = np.mean(etas)
        p50 = np.percentile(etas, 50)
        p90 = np.percentile(etas, 90)
        utilization = [c.total_work_time / total_sim_time for c in self.couriers]
        queued_orders = len(self.queued_orders)
        queued_ratio = queued_orders / len(self.orders)

        return{
            "avg_eta": avg_eta,
            "p50": p50,
            "p90": p90,
            "utilization": utilization,
            "queued_orders": len(self.queued_orders),
            "queued_ratio": queued_ratio
        }

distance_lookup = distance_df.set_index(["PULocationID", "DOLocationID"])["great_circle_km"].to_dict()
speed_lookup = (
    speed_df.groupby(["PULocationID", "DOLocationID", "pickup_hour"])["historical_speed_kmh"]
    .median()
    .to_dict()
)

def predict_eta(order, current_time, courier_zone):
    pu = order.pickup_zone
    do = order.dropoff_zone
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
    return [Courier(courier_id=i, current_zone=random.choice(zone_ids), available_at=0.0) for i in range(n)]

def generate_fake_orders(n: int, zone_ids: List[int], start_time=0.0, interval=60.0) -> List[Order]:
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
    NUM_COURIERS = 2
    NUM_ORDERS = 10
    INTERVAL = 10.0

    zone_ids = list(range(1, 264))
    couriers = initialize_couriers(NUM_COURIERS, zone_ids)
    orders = generate_fake_orders(NUM_ORDERS, zone_ids, interval=INTERVAL)

    sim = DispatchSimulator(couriers, orders, predict_eta)
    sim.run()

    print("\nAssignments:")
    for a in sim.assignments:
        print(f"Order {a[0]} assigned to Courier {a[1]} with total ETA {a[2]:.1f} sec")

    sim.report_metrics()