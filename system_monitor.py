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

# import psutil
# import pandas as pd
# import customtkinter as ctk
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import time

# class SystemMonitor:
#     def __init__(self):
#         self.data = pd.DataFrame(columns=['timestamp', 'cpu', 'ram', 'disk'])
#         self.monitoring = False
#         self.max_history = 60  # Keep last 60 seconds of data

#     def collect_metrics(self):
#         """Collect real-time system metrics."""
#         cpu = psutil.cpu_percent(interval=1)
#         ram = psutil.virtual_memory().percent
#         disk = psutil.disk_usage('/').percent
#         timestamp = time.time()
#         return {'timestamp': timestamp, 'cpu': cpu, 'ram': ram, 'disk': disk}

#     def monitor_system(self):
#         """Update dataset with latest metrics."""
#         metrics = self.collect_metrics()
#         self.data = pd.concat([self.data, pd.DataFrame([metrics])], ignore_index=True)

#         # Keep only the last `max_history` seconds
#         if len(self.data) > self.max_history:
#             self.data = self.data.iloc[-self.max_history:]

#         return metrics

#     def get_process_data(self):
#         """Collect CPU usage of all processes."""
#         processes = []
#         for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
#             try:
#                 processes.append(proc.info)
#             except (psutil.NoSuchProcess, psutil.AccessDenied):
#                 continue
#         return pd.DataFrame(processes)

#     def display_gui(self):
#         ctk.set_appearance_mode("dark")
#         root = ctk.CTk()
#         root.title("Real-Time System Monitor")
#         root.geometry("1000x800")

#         # Metrics frame
#         metrics_frame = ctk.CTkFrame(root)
#         metrics_frame.pack(pady=10, padx=10, fill="x")
#         cpu_label = ctk.CTkLabel(metrics_frame, text="CPU: 0%")
#         cpu_label.pack(anchor="w")
#         ram_label = ctk.CTkLabel(metrics_frame, text="RAM: 0%")
#         ram_label.pack(anchor="w")
#         disk_label = ctk.CTkLabel(metrics_frame, text="Disk: 0%")
#         disk_label.pack(anchor="w")

#         # Plot frame (system usage)
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
#         fig.tight_layout(pad=4.0)

#         canvas = FigureCanvasTkAgg(fig, master=root)
#         canvas.get_tk_widget().pack(pady=10, padx=10, fill="both", expand=True)

#         # Control frame
#         control_frame = ctk.CTkFrame(root)
#         control_frame.pack(pady=10, padx=10, fill="x")
#         toggle_button = ctk.CTkButton(control_frame, text="Stop Monitoring",
#                                       command=lambda: self.toggle_monitoring(root))
#         toggle_button.pack()

#         def update_gui():
#             if not self.monitoring:
#                 return

#             # ---- System metrics ----
#             metrics = self.monitor_system()
#             cpu_label.configure(text=f"CPU: {metrics['cpu']:.1f}%")
#             ram_label.configure(text=f"RAM: {metrics['ram']:.1f}%")
#             disk_label.configure(text=f"Disk: {metrics['disk']:.1f}%")

#             ax1.clear()
#             times = (self.data['timestamp'] - self.data['timestamp'].iloc[-1]).values
#             ax1.plot(times, self.data['cpu'], label="CPU")
#             ax1.plot(times, self.data['ram'], label="RAM")
#             ax1.plot(times, self.data['disk'], label="Disk")
#             ax1.set_title("System Resource Usage (last 60s)")
#             ax1.set_xlabel("Time (s ago)")
#             ax1.set_ylabel("Usage (%)")
#             ax1.legend()

#             # ---- Process scatter plot ----
#             ax2.clear()
#             process_df = self.get_process_data()
#             if not process_df.empty:
#                 ax2.scatter(process_df['pid'], process_df['cpu_percent'], alpha=0.6)
#                 ax2.set_title("CPU Usage by Processes")
#                 ax2.set_xlabel("Process ID (PID)")
#                 ax2.set_ylabel("CPU Usage (%)")

#                 # Annotate top 5 CPU-hogging processes
#                 top_procs = process_df.nlargest(5, 'cpu_percent')
#                 for _, row in top_procs.iterrows():
#                     ax2.annotate(row['name'], (row['pid'], row['cpu_percent']),
#                                  textcoords="offset points", xytext=(5,5), fontsize=8)

#             canvas.draw()
#             root.after(2000, update_gui)  # update every 2 seconds

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
        """Collect CPU usage of all processes (with usernames)."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'username']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        df = pd.DataFrame(processes)

        if not df.empty:
            max_cpu = df['cpu_percent'].max()
            if max_cpu > 0:
                df['relative_cpu_percent'] = (df['cpu_percent'] / max_cpu) * 100
            else:
                df['relative_cpu_percent'] = 0
        return df
    
    def display_gui(self):
        ctk.set_appearance_mode("dark")
        root = ctk.CTk()
        root.title("Responsive Real-Time System Monitor")
        root.geometry("1000x800")

        # Responsive grid for the root window
        root.grid_rowconfigure(0, weight=0) # For metrics
        root.grid_rowconfigure(1, weight=1) # For tabview
        root.grid_columnconfigure(0, weight=1)

        # --- Metrics Frame (Top) ---
        metrics_frame = ctk.CTkFrame(root, height=50)
        metrics_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        metrics_frame.pack_propagate(False) # Prevent frame from resizing to content

        # Metric labels
        cpu_label = ctk.CTkLabel(metrics_frame, text="CPU: 0%", font=("Arial", 18))
        cpu_label.pack(side="left", padx=20, pady=5)
        ram_label = ctk.CTkLabel(metrics_frame, text="RAM: 0%", font=("Arial", 18))
        ram_label.pack(side="left", padx=20, pady=5)
        disk_label = ctk.CTkLabel(metrics_frame, text="DISK: 0%", font=("Arial", 18))
        disk_label.pack(side="left", padx=20, pady=5)

        # --- Tabview for Plots ---
        tabview = ctk.CTkTabview(root, fg_color="#F0F0F0",
                                segmented_button_fg_color=("gray60", "gray30"),
                                segmented_button_selected_color="gray70",
                                segmented_button_selected_hover_color="gray70",
                                segmented_button_unselected_color="gray20",
                                segmented_button_unselected_hover_color="gray20")
        tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # --- System Usage Tab ---
        system_usage_tab = tabview.add("SYSTEM USAGE")
        system_usage_tab.grid_rowconfigure(0, weight=1)
        system_usage_tab.grid_columnconfigure(0, weight=1)

        line_plot_frame = ctk.CTkFrame(system_usage_tab, fg_color="white")
        line_plot_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        line_plot_frame.grid_rowconfigure(0, weight=1)
        line_plot_frame.grid_columnconfigure(0, weight=1)

        line_fig, line_ax = plt.subplots(figsize=(8, 6), facecolor="white")
        line_ax.set_facecolor("#F0F0F0")
        line_canvas = FigureCanvasTkAgg(line_fig, master=line_plot_frame)
        line_canvas_widget = line_canvas.get_tk_widget()
        line_canvas_widget.pack(expand=True, fill="both")

        # --- Scatter Plot Tab ---
        scatter_plot_tab = tabview.add("SCATTER PLOT")
        scatter_plot_tab.grid_rowconfigure(0, weight=1)
        scatter_plot_tab.grid_columnconfigure(0, weight=1)

        scatter_plot_frame = ctk.CTkFrame(scatter_plot_tab, fg_color="white")
        scatter_plot_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        scatter_plot_frame.grid_rowconfigure(0, weight=1)
        scatter_plot_frame.grid_columnconfigure(0, weight=1)

        scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6), facecolor="white")
        scatter_ax.set_facecolor("#F0F0F0")
        scatter_canvas = FigureCanvasTkAgg(scatter_fig, master=scatter_plot_frame)
        scatter_canvas_widget = scatter_canvas.get_tk_widget()
        scatter_canvas_widget.pack(expand=True, fill="both")
        
        tabview.set("SYSTEM USAGE")

        def update_gui():
            if not self.monitoring:
                return

            # ---- System metrics ----
            metrics = self.monitor_system()
            cpu_label.configure(text=f"CPU: {metrics['cpu']:.1f}%")
            ram_label.configure(text=f"RAM: {metrics['ram']:.1f}%")
            disk_label.configure(text=f"DISK: {metrics['disk']:.1f}%")

            # ---- Historical usage plot (Line Graph) ----
            line_ax.clear()
            times = (self.data['timestamp'] - self.data['timestamp'].iloc[-1]).values if not self.data.empty else []
            line_ax.plot(times, self.data['cpu'], color='darkred', label="CPU")
            line_ax.plot(times, self.data['ram'], color='green', label="RAM")
            line_ax.plot(times, self.data['disk'], color='blue', label="DISK")
            
            line_ax.set_title("")
            line_ax.set_xlabel("Timeline")
            line_ax.set_ylabel("Usage (%)")
            line_ax.set_ylim(0, 100)
            line_ax.legend()
            line_fig.tight_layout(pad=3.0)
            line_canvas.draw()

            # ---- User/Process Scatter Plot ----
            scatter_ax.clear()
            process_df = self.get_process_data()
            
            if not process_df.empty:
                user_cpu_summary = process_df.groupby('username')['cpu_percent'].sum().reset_index()
                user_threshold = 1.0 
                relevant_users = user_cpu_summary[user_cpu_summary['cpu_percent'] > user_threshold]['username'].tolist()

                filtered_df = process_df[process_df['username'].isin(relevant_users)].copy()
                
                process_threshold = 0.5 
                filtered_df = filtered_df[filtered_df['cpu_percent'] >= process_threshold]

                if not filtered_df.empty:
                    filtered_df = filtered_df.sort_values(by=['username', 'cpu_percent'], ascending=[True, False]).reset_index(drop=True)

                    users = filtered_df['username'].unique()
                    x_vals = []
                    y_vals = []
                    x_tick_labels = []
                    process_names = [] # Store process names for annotation
                    
                    blue_shade_1 = '#ADD8E6' 
                    blue_shade_2 = '#87CEEB' 

                    for rect in scatter_ax.patches:
                        rect.remove()

                    current_x_offset = 0
                    for i, user in enumerate(users):
                        user_procs = filtered_df[filtered_df['username'] == user]
                        
                        user_x_indices = list(range(current_x_offset, current_x_offset + len(user_procs)))
                        
                        x_vals.extend(user_x_indices)
                        y_vals.extend(user_procs['cpu_percent'].tolist())
                        process_names.extend(user_procs['name'].tolist()) # Collect process names
                        
                        if user_procs.empty:
                            continue
                        
                        center_x_pos = current_x_offset + (len(user_procs) - 1) / 2
                        x_tick_labels.append((center_x_pos, user))

                        current_x_offset += len(user_procs) + 1 

                    scatter_ax.scatter(x_vals, y_vals, c='blue', alpha=0.7)
                    
                    # --- NEW: Add process name annotations ---
                    for j, (x, y) in enumerate(zip(x_vals, y_vals)):
                        # Shorten process name if too long, or use '...'
                        name = process_names[j]
                        if len(name) > 15: # Arbitrary length to keep labels concise
                            name = name[:12] + '...'
                        scatter_ax.annotate(name, 
                                            (x, y), 
                                            textcoords="offset points", # how to position the text
                                            xytext=(0,5),               # distance from text to points (x,y)
                                            ha='center',                # horizontal alignment
                                            va='bottom',                # vertical alignment
                                            fontsize=7,                 # small but visible font size
                                            color='dimgray',            # color for visibility
                                            alpha=0.8)                  # slight transparency
                    # --- END NEW ---

                    max_cpu = max(y_vals) if y_vals else 0
                    scatter_ax.set_ylim(0, max_cpu * 1.2 if max_cpu > 0 else 100)
                    
                    current_x_patch_start = 0
                    for i, user in enumerate(users):
                        user_procs_count = len(filtered_df[filtered_df['username'] == user])
                        
                        rect_color = blue_shade_1 if i % 2 == 0 else blue_shade_2
                        
                        rect = plt.Rectangle((current_x_patch_start - 0.5, scatter_ax.get_ylim()[0]),
                                            user_procs_count + 1,
                                            scatter_ax.get_ylim()[1] - scatter_ax.get_ylim()[0],
                                            facecolor=rect_color, edgecolor='none', zorder=0)
                        scatter_ax.add_patch(rect)
                        current_x_patch_start += user_procs_count + 1

                    tick_positions = [pos for pos, _ in x_tick_labels]
                    tick_labels_text = [label for _, label in x_tick_labels]
                    
                    scatter_ax.set_xticks(tick_positions)
                    scatter_ax.set_xticklabels(tick_labels_text, rotation=45, ha="right", fontsize=8)
                    
                    scatter_ax.set_title("")
                    scatter_ax.set_xlabel("User / Process Category")
                    scatter_ax.set_ylabel("CPU Usage (%)")
                    scatter_ax.tick_params(axis='x', length=0)
            
            scatter_fig.tight_layout(pad=3.0)
            scatter_canvas.draw()

            root.after(2000, update_gui)

        self.monitoring = True
        update_gui()
        root.mainloop()

        
    def toggle_monitoring(self, root):
        self.monitoring = not self.monitoring
        root.destroy() if not self.monitoring else None


if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.display_gui()