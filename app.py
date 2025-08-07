import streamlit as st
import time
import pandas as pd
import pydeck as pdk

from src.utils.geo import load_zone_latlons

from src.dispatch.simulator import(
    initialize_couriers,
    generate_fake_orders,
    DispatchSimulator,
    predict_eta,
)

st.title("ETA Dispatch Simulator")
zone_coords = load_zone_latlons()

sidebar_tabs = st.sidebar.tabs(["⚙️ Simulation Settings", "🧩 Display Options"])
with sidebar_tabs[0]:
    num_couriers = st.slider("Number of Couriers", 1, 50, 5)
    num_orders = st.slider("Number of Orders", 1, 100, 20)
    order_interval = st.slider("Order Interval (sec)", 10, 300, 60)

if st.sidebar.button("🔁 Rerun Simulation"):
    st.session_state["last_run"] = 0
    st.rerun()

with sidebar_tabs[1]:
    show_only_assigned = st.checkbox("Show Only Assigned Couriers", value=False)

courier_colors = {}

run_simulation = st.session_state.get("last_run", 0) + 0.3 < time.time()
if run_simulation:
    st.session_state["last_run"] = time.time()
    with st.spinner("⏳ Initializing couriers, generating orders, and running simulation..."):
        time.sleep(1)
        zone_ids = list(range(1, 264))
        couriers = initialize_couriers(num_couriers, zone_ids)
        orders = generate_fake_orders(num_orders, zone_ids, interval=order_interval)
        
        sim = DispatchSimulator(couriers, orders, predict_eta)
        sim.run()
        assigned_couriers = set(courier_id for _, courier_id, _ in sim.assignments)
        # Generate unique colors for couriers
        import random
        random.seed(42)
        courier_colors = {
            c.courier_id: [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
            for c in sim.couriers
        }

        map_points = []

        # Add courier markers
        for c in sim.couriers:
            if show_only_assigned and c.courier_id not in assigned_couriers:
                continue
            if c.current_zone in zone_coords:
                lat, lon = zone_coords[c.current_zone]
                map_points.append({"lat": lat, "lon": lon, "color": courier_colors[c.courier_id], "label": f"Courier {c.courier_id + 1}"})

        # Add order pickup markers with same color as their assigned courier
        for order_id, courier_id, _ in sim.assignments:
            pickup_zone = sim.orders[order_id].pickup_zone
            if pickup_zone in zone_coords:
                lat, lon = zone_coords[pickup_zone]
                map_points.append({"lat": lat, "lon": lon, "color": courier_colors[courier_id], "label": f"Order {order_id + 1}"})

        map_df = pd.DataFrame(map_points)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=120,
            pickable=True,
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
            get_radius_scale=1,
        )

        view_state = pdk.ViewState(
            latitude=map_df["lat"].mean(),
            longitude=map_df["lon"].mean(),
            zoom=10,
            pitch=0,
        )
    tabs = st.tabs(["🗺️ Map", "📊 Metrics", "📋 Assignments"])

    with tabs[0]:
        st.markdown("## 🗺️ Courier and Order Locations")
        st.divider()
        tooltip = {"text": "{label}"}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

    with tabs[1]:
        st.markdown("## 📊 Metrics")
        st.divider()
        metrics = sim.report_metrics()
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg ETA", f"{metrics['avg_eta']:.1f} sec")
            col2.metric("P50 ETA", f"{metrics['p50']:.1f} sec")
            col3.metric("P90 ETA", f"{metrics['p90']:.1f} sec")
            st.metric("Queued Orders", f"{metrics['queued_orders']} ({metrics['queued_ratio']*100:.1f}%)")
            unassigned_count = len(sim.orders) - len(sim.assignments)
            st.metric("Total Orders", f"{len(sim.orders)}")
            st.metric("Unassigned Orders", f"{unassigned_count} ({(unassigned_count/len(sim.orders))*100:.1f}%)")

            assign_df = pd.DataFrame(sim.assignments, columns=["Order ID", "Courier ID", "ETA (sec)"])
            assign_df["Order ID"] += 1
            assign_df["Courier ID"] += 1
            st.download_button("📥 Download Assignments as CSV", assign_df.to_csv(index=False), file_name="assignments.csv", mime="text/csv")

            st.markdown("### 🛵 Courier Utilization")
            for idx, u in enumerate(metrics['utilization']):
                color = courier_colors.get(idx, [0, 0, 0])
                hex_color = '#%02x%02x%02x' % tuple(color)
                st.markdown(f"<span style='color:{hex_color}; font-weight:bold'>Courier {idx + 1}</span>: {u*100:.1f}%", unsafe_allow_html=True)
        else:
            st.warning("No metrics available.")

    with tabs[2]:
        st.markdown("## 📋 Assignments by Courier")
        grouped_assignments = {}
        for order_id, courier_id, eta in sim.assignments:
            if courier_id not in grouped_assignments:
                grouped_assignments[courier_id] = []
            grouped_assignments[courier_id].append((order_id, eta))


        for courier_id, assignments in grouped_assignments.items():
            color = courier_colors[courier_id]
            hex_color = '#%02x%02x%02x' % tuple(color)
            st.markdown(f"### 🚴 <span style='color:{hex_color}; font-weight:bold'>Courier {courier_id + 1}</span>", unsafe_allow_html=True)
            for order_id, eta in assignments:
                st.markdown(f"- Order {order_id + 1} | ETA: {eta:.1f} sec")

        if sim.assignments:
            st.markdown("## 📈 Dispatch Insights")
            st.divider()

            most_loaded = max(grouped_assignments.items(), key=lambda x: len(x[1]))
            st.markdown(f"**Most Assigned Courier:** Courier {most_loaded[0] + 1} with {len(most_loaded[1])} orders.")

            max_eta_assignment = max(sim.assignments, key=lambda x: x[2])
            order_id, courier_id, eta = max_eta_assignment
            st.markdown(f"**Longest ETA:** Order {order_id + 1} assigned to Courier {courier_id + 1} with ETA {eta:.1f} sec.")

            unassigned_count = len(sim.orders) - len(sim.assignments)
            st.markdown(f"**Unassigned Orders:** {unassigned_count}")
        else:
            st.warning("No assignments made.")