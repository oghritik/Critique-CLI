# import psutil
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# import customtkinter as ctk
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import threading
# import time

# class SystemMonitor:
#     def __init__(self):
#         self.data = pd.DataFrame(columns=['timestamp', 'cpu', 'ram', 'disk'])
#         self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#         self.monitoring = False
#         self.window_size = 5  # For moving average
#         self.max_history = 60  # Store 60 seconds of data
#         self.labels = ['low', 'normal', 'extreme']
#         self.train_classifier()

#     def train_classifier(self):
#         # Synthetic training data based on thresholds
#         X = []
#         y = []
#         for usage in range(0, 100, 5):
#             for delta in range(-20, 21, 5):
#                 X.append([usage, delta])
#                 if usage > 80 or delta > 20:
#                     y.append(2)  # Extreme
#                 elif usage < 20:
#                     y.append(0)  # Low
#                 else:
#                     y.append(1)  # Normal
#         self.classifier.fit(X, y)

#     def collect_metrics(self):
#         cpu = psutil.cpu_percent(interval=1)
#         ram = psutil.virtual_memory().percent
#         disk = psutil.disk_usage('/').percent
#         timestamp = time.time()
#         return {'timestamp': timestamp, 'cpu': cpu, 'ram': ram, 'disk': disk}

#     def predict_usage(self, series):
#         # Simple moving average for forecasting
#         if len(series) < self.window_size:
#             return series[-1] if len(series) > 0 else 0
#         return np.mean(series[-self.window_size:])

#     def classify_usage(self, usage, delta):
#         return self.labels[self.classifier.predict([[usage, delta]])[0]]

#     def monitor_system(self):
#         if len(self.data) > 0:
#             deltas = self.data[['cpu', 'ram', 'disk']].diff().iloc[-1].fillna(0)
#         else:
#             deltas = {'cpu': 0, 'ram': 0, 'disk': 0}
#         metrics = self.collect_metrics()
#         self.data = pd.concat([self.data, pd.DataFrame([metrics])], ignore_index=True)
#         if len(self.data) > self.max_history:
#             self.data = self.data.iloc[-self.max_history:]

#         # Predict next 10 seconds
#         predictions = {}
#         for metric in ['cpu', 'ram', 'disk']:
#             series = self.data[metric].values
#             predictions[metric] = [self.predict_usage(series) for _ in range(10)]

#         # Classify current usage
#         classifications = {}
#         for metric in ['cpu', 'ram', 'disk']:
#             classifications[metric] = self.classify_usage(metrics[metric], deltas[metric])

#         return metrics, predictions, classifications

#     def display_gui(self):
#         ctk.set_appearance_mode("dark")
#         root = ctk.CTk()
#         root.title("System Monitor")
#         root.geometry("800x600")

#         # Metrics frame
#         metrics_frame = ctk.CTkFrame(root)
#         metrics_frame.pack(pady=10, padx=10, fill="x")
#         cpu_label = ctk.CTkLabel(metrics_frame, text="CPU: 0% (Unknown)")
#         cpu_label.pack(anchor="w")
#         ram_label = ctk.CTkLabel(metrics_frame, text="RAM: 0% (Unknown)")
#         ram_label.pack(anchor="w")
#         disk_label = ctk.CTkLabel(metrics_frame, text="Disk: 0% (Unknown)")
#         disk_label.pack(anchor="w")

#         # Plot frame
#         fig, ax = plt.subplots(figsize=(6, 4))
#         ax.set_title("System Resource Usage")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Usage (%)")
#         canvas = FigureCanvasTkAgg(fig, master=root)
#         canvas.get_tk_widget().pack(pady=10, padx=10, fill="both", expand=True)

#         # Control frame
#         control_frame = ctk.CTkFrame(root)
#         control_frame.pack(pady=10, padx=10, fill="x")
#         toggle_button = ctk.CTkButton(control_frame, text="Stop Monitoring", command=lambda: self.toggle_monitoring(root))
#         toggle_button.pack()

#         def update_gui():
#             if not self.monitoring:
#                 return
#             metrics, predictions, classifications = self.monitor_system()
#             cpu_label.configure(text=f"CPU: {metrics['cpu']:.1f}% ({classifications['cpu'].capitalize()})")
#             ram_label.configure(text=f"RAM: {metrics['ram']:.1f}% ({classifications['ram'].capitalize()})")
#             disk_label.configure(text=f"Disk: {metrics['disk']:.1f}% ({classifications['disk'].capitalize()})")

#             ax.clear()
#             times = (self.data['timestamp'] - self.data['timestamp'].iloc[-1]).values
#             future_times = np.arange(1, 11)
#             for metric in ['cpu', 'ram', 'disk']:
#                 ax.plot(times, self.data[metric], label=f"{metric.capitalize()} (Historical)")
#                 ax.plot(future_times, predictions[metric], '--', label=f"{metric.capitalize()} (Predicted)")
#             ax.set_title("System Resource Usage")
#             ax.set_xlabel("Time (s)")
#             ax.set_ylabel("Usage (%)")
#             ax.legend()
#             canvas.draw()

#             if self.monitoring:
#                 root.after(1000, update_gui)

#         self.monitoring = True
#         update_gui()
#         root.mainloop()

#     def toggle_monitoring(self, root):
#         self.monitoring = not self.monitoring
#         root.destroy() if not self.monitoring else None
        
# if __name__ == "__main__":
#     monitor = SystemMonitor()
#     monitor.display_gui()
#--------------------------------------------------------------------------------
import psutil
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class SystemMonitor:
    def __init__(self):
        self.data = pd.DataFrame(columns=['timestamp', 'cpu', 'ram', 'disk'])
        self.monitoring = False
        self.max_history = 60  # Keep last 60 seconds of data

    def collect_metrics(self):
        """Collect real-time system metrics."""
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        timestamp = time.time()
        return {'timestamp': timestamp, 'cpu': cpu, 'ram': ram, 'disk': disk}

    def monitor_system(self):
        """Update dataset with latest metrics."""
        metrics = self.collect_metrics()
        self.data = pd.concat([self.data, pd.DataFrame([metrics])], ignore_index=True)

        # Keep only the last `max_history` seconds
        if len(self.data) > self.max_history:
            self.data = self.data.iloc[-self.max_history:]

        return metrics

    def get_process_data(self):
        """Collect CPU usage of all processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pd.DataFrame(processes)

    def display_gui(self):
        ctk.set_appearance_mode("dark")
        root = ctk.CTk()
        root.title("Real-Time System Monitor")
        root.geometry("1000x800")

        # Metrics frame
        metrics_frame = ctk.CTkFrame(root)
        metrics_frame.pack(pady=10, padx=10, fill="x")
        cpu_label = ctk.CTkLabel(metrics_frame, text="CPU: 0%")
        cpu_label.pack(anchor="w")
        ram_label = ctk.CTkLabel(metrics_frame, text="RAM: 0%")
        ram_label.pack(anchor="w")
        disk_label = ctk.CTkLabel(metrics_frame, text="Disk: 0%")
        disk_label.pack(anchor="w")

        # Plot frame (system usage)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        fig.tight_layout(pad=4.0)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(pady=10, padx=10, fill="both", expand=True)

        # Control frame
        control_frame = ctk.CTkFrame(root)
        control_frame.pack(pady=10, padx=10, fill="x")
        toggle_button = ctk.CTkButton(control_frame, text="Stop Monitoring",
                                      command=lambda: self.toggle_monitoring(root))
        toggle_button.pack()

        def update_gui():
            if not self.monitoring:
                return

            # ---- System metrics ----
            metrics = self.monitor_system()
            cpu_label.configure(text=f"CPU: {metrics['cpu']:.1f}%")
            ram_label.configure(text=f"RAM: {metrics['ram']:.1f}%")
            disk_label.configure(text=f"Disk: {metrics['disk']:.1f}%")

            ax1.clear()
            times = (self.data['timestamp'] - self.data['timestamp'].iloc[-1]).values
            ax1.plot(times, self.data['cpu'], label="CPU")
            ax1.plot(times, self.data['ram'], label="RAM")
            ax1.plot(times, self.data['disk'], label="Disk")
            ax1.set_title("System Resource Usage (last 60s)")
            ax1.set_xlabel("Time (s ago)")
            ax1.set_ylabel("Usage (%)")
            ax1.legend()

            # ---- Process scatter plot ----
            ax2.clear()
            process_df = self.get_process_data()
            if not process_df.empty:
                ax2.scatter(process_df['pid'], process_df['cpu_percent'], alpha=0.6)
                ax2.set_title("CPU Usage by Processes")
                ax2.set_xlabel("Process ID (PID)")
                ax2.set_ylabel("CPU Usage (%)")

                # Annotate top 5 CPU-hogging processes
                top_procs = process_df.nlargest(5, 'cpu_percent')
                for _, row in top_procs.iterrows():
                    ax2.annotate(row['name'], (row['pid'], row['cpu_percent']),
                                 textcoords="offset points", xytext=(5,5), fontsize=8)

            canvas.draw()
            root.after(2000, update_gui)  # update every 2 seconds

        self.monitoring = True
        update_gui()
        root.mainloop()

    def toggle_monitoring(self, root):
        self.monitoring = not self.monitoring
        root.destroy() if not self.monitoring else None

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.display_gui()