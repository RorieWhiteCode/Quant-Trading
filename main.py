
import matplotlib.pyplot as plt
import numpy as np
import requests
import time
import time
from playsound import playsound
import tkinter as tk
import threading
import plotly.graph_objs as go
def main():
    print("Choose a mode to run:")
    print("1. Dashboard Mode (default)")
    print("2. Simulation Mode")
    mode = input("Enter 1 or 2: ")
    if mode not in ["1", "2"]:
        mode = "1"

    if mode == "1":
        print("Running Dashboard Mode...")
        mean_buy = 3278.98  # Your mean buy price
        liquidation = 3204.94  # Your liquidation price
        TP = CONFIG['take_profit']
        SL = CONFIG['stop_loss']
        last_alert_time = 0  # To manage alert frequency
        alert_interval = 60  # Minimum time between alerts in seconds

        print("Starting Dynamic Quant Trading System...\n")

        plt.ion()  # Enable interactive mode for dynamic graphing
        last_update_time = time.time()  # Initialize the last update timestamp

        while True:
            current_time = time.time()
            if current_time - last_update_time >= CONFIG['update_interval']:
                last_update_time = current_time  # Reset the last update time

                # Fetch live price and dynamic parameters
                P0 = fetch_live_price()
                if P0 is None:
                    print("Error fetching live price. Skipping this update.\n", flush=True)
                    continue

                # Trigger breakout alert
                breakout_alert(P0, TP, breakout_threshold=0.01)  # Alert when price is within 1% of TP

                sigma = fetch_volatility()
                mu = calculate_drift()

                # Run Monte Carlo simulation
                price_paths = run_monte_carlo(P0, mu, sigma)
                time_steps = np.linspace(0, CONFIG['time_horizon'], CONFIG['time_horizon'] + 1)

                # Calculate probabilities
                hits_tp = np.sum(np.any(price_paths >= TP, axis=1)) / CONFIG['simulations']
                hits_sl = np.sum(np.any(price_paths <= SL, axis=1)) / CONFIG['simulations']

                # Calculate dynamic RRR
                rrr = calculate_dynamic_rrr(hits_tp, hits_sl, mean_buy, liquidation, TP)

                # Check for Take Profit condition
                if P0 >= TP:
                    print(f"Take Profit Hit: ${P0:.2f}")
                    flash_screen_on_tp()  # Trigger the flashing screen alert

                # Generate and log sell alert
                alert = generate_sell_alert(P0, TP, SL, liquidation, rrr)
                if alert:
                    print(alert, flush=True)
                    # Ensure alerts are not too frequent
                    if current_time - last_alert_time >= alert_interval:
                        play_alert_sound()  # Play beep
                        last_alert_time = current_time  # Update last alert time
                else:
                    print(f"INFO: Hold position. Current price (${P0:.2f}) is within safe range.", flush=True)

                # Count and visualize distributions
                hits_tp_count, hits_sl_count, neutral_count = count_and_visualize_distributions(price_paths, TP, SL)

                # Display results
                print(f"Live Price: ${P0:.2f}", flush=True)
                print(f"Probability of Hitting TP: {hits_tp * 100:.2f}%", flush=True)
                print(f"Probability of Hitting SL: {hits_sl * 100:.2f}%", flush=True)
                print(f"Dynamic RRR: {rrr:.2f}", flush=True)
                print(f"Liquidation Price: ${liquidation:.2f}\n", flush=True)

                # Plot dynamic graph
                plot_dynamic_graph(price_paths, TP, SL, time_steps, hits_tp, hits_sl, liquidation)

            # Pause to reduce CPU usage
            time.sleep(0.5)

    elif mode == "2":
        print("Running Simulation Mode...")
        
        # Simulation parameters
        initial_price = 0  # Starting price of Dogecoin
        mu = 0  # Drift (2% daily return)
        sigma = 0  # Volatility (10%)
        days = 0  # Simulate over 3 days
        steps_per_day = 0  # Hourly intervals
        simulations = 0  # Number of paths

        # Function to simulate paths
        def simulate_doge_paths(initial_price, mu, sigma, days, steps_per_day, simulations):
            dt = 1 / steps_per_day
            total_steps = days * steps_per_day
            paths = np.zeros((simulations, total_steps + 1))
            paths[:, 0] = initial_price

            for t in range(1, total_steps + 1):
                random_shocks = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=simulations)
                paths[:, t] = paths[:, t - 1] * np.exp(random_shocks)

            return paths

        # Simulate Dogecoin paths
        doge_paths = simulate_doge_paths(initial_price, mu, sigma, days, steps_per_day, simulations)
        time_steps = np.linspace(0, days, days * steps_per_day + 1)

        # Visualize the paths
        fig = go.Figure()
        for path in doge_paths[:100]:
            fig.add_trace(go.Scatter(x=time_steps, y=path, mode='lines', line=dict(width=1), opacity=0.2))

        fig.update_layout(
            title='Simulated Dogecoin Price Paths Over 3 Days',
            xaxis_title='Time (Days)',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            paper_bgcolor='#121212',
            plot_bgcolor='#1e1e1e'
        )
        fig.show()
if __name__ == "__main__":
    main()
