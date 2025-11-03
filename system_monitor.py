import sys
import os
import psutil
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.ticker import FormatStrFormatter

from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QStackedWidget, QFrame
)
from PySide6.QtCore import Qt, QTimer, QThread, QObject, Signal, Slot
from PySide6.QtGui import QFont, QColor

# --- Force Matplotlib Backend ---
# This is the correct backend for PySide6/PyQt
os.environ['MPLBACKEND'] = 'QtAgg'
import matplotlib
matplotlib.use('QtAgg', force=True)

# --- Matplotlib Style (Your settings) ---
# We set the facecolor to transparent so it shows the
# rounded-corner widget background
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'figure.figsize': (8, 4),
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'lines.linewidth': 2.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    
    # Key change for PySide6 integration:
    'figure.facecolor': 'none', 
    'axes.facecolor': 'none', 
    'savefig.facecolor': 'none',
})


# ==============================================================================
# STEP 1: THE DATA WORKER (MOVED TO A QTHREAD)
# ==============================================================================

class SystemDataWorker(QObject):
    """
    Runs in a separate thread to collect system data without
    blocking the main GUI.
    """
    # Define signals that will carry the new data
    dataUpdated = Signal(dict, pd.DataFrame)
    processDataUpdated = Signal(pd.DataFrame)
    
    def __init__(self, max_history=60):
        super().__init__()
        self.data = pd.DataFrame(columns=['timestamp', 'cpu', 'ram', 'disk'])
        self.max_history = max_history
        self.monitoring = False

    def start_monitoring(self):
        """Starts the monitoring loop."""
        self.monitoring = True
        
        # --- Timer to control update frequency ---
        # We use a QTimer *inside* the thread for stable timing
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_monitor_cycle)
        self.timer.start(500) # Update every 500ms

    def stop_monitoring(self):
        """Stops the monitoring loop."""
        self.monitoring = False
        if hasattr(self, 'timer'):
            self.timer.stop()

    def run_monitor_cycle(self):
        """
        This is the main loop of the worker. It collects data
        and emits signals.
        """
        if not self.monitoring:
            return
            
        try:
            # 1. Collect system metrics
            metrics = self.collect_metrics()
            timestamp = time.time()
            new_data = {'timestamp': timestamp, **metrics}

            if self.data.empty:
                self.data = pd.DataFrame([new_data])
            else:
                new_row = pd.DataFrame([new_data])
                self.data = pd.concat([self.data, new_row], ignore_index=True)
                
            if len(self.data) > self.max_history:
                self.data = self.data.iloc[-self.max_history:]
            
            # EMIT SIGNAL 1
            self.dataUpdated.emit(metrics, self.data.copy())

            # 2. Collect process data
            process_df = self.get_process_data()
            
            # EMIT SIGNAL 2
            self.processDataUpdated.emit(process_df)
            
        except Exception as e:
            print(f"Monitoring error: {e}")

    # --- Your data collection methods, unchanged ---
    def collect_metrics(self):
        """Collect real-time system metrics."""
        cpu = psutil.cpu_percent(interval=None) # No interval for non-blocking
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        return {'cpu': cpu, 'ram': ram, 'disk': disk}
    
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


# ==============================================================================
# STEP 2: THE PYSIXDE6 GUI SHELL
# ==============================================================================

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.display_time_window = 30 # From your original code
        
        # --- Fonts ---
        self.nav_font_bold = QFont("Arial", 15, QFont.Weight.Bold)
        self.nav_font_normal = QFont("Arial", 15)
        self.legend_font = QFont("Arial", 16)
        
        self.init_ui()
        self.init_worker_thread()
        
        # Set initial view
        self.switch_view(0)

    def init_ui(self):
        """Create the main GUI layout."""
        
        self.setWindowTitle("System Monitor (PySide6)")
        self.setGeometry(100, 100, 700, 450)
        self.setMinimumSize(700, 450)
        
        # Set main window background color
        self.setStyleSheet("background-color: #212121;")

        # --- Main Central Widget & Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Custom Navigation Header ---
        nav_frame = QFrame(self)
        nav_frame.setFixedHeight(60)
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(25, 15, 25, 15)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.usage_button = QPushButton("SYSTEM USAGE")
        self.scatter_button = QPushButton("SCATTER PLOT")
        
        # Connect buttons to the switch_view slot
        self.usage_button.clicked.connect(lambda: self.switch_view(0))
        self.scatter_button.clicked.connect(lambda: self.switch_view(1))

        # Apply stylesheet for buttons (replaces configure)
        button_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 0;
                margin-right: 20px;
            }
        """
        self.usage_button.setStyleSheet(button_style)
        self.scatter_button.setStyleSheet(button_style)

        nav_layout.addWidget(self.usage_button)
        nav_layout.addWidget(self.scatter_button)
        
        main_layout.addWidget(nav_frame)

        # --- Main Content Area (QStackedWidget) ---
        # This is the proper way to switch between views
        self.plot_stack = QStackedWidget()
        main_layout.addWidget(self.plot_stack)
        
        # --- Plot 1: Line Plot ---
        self.line_plot_widget = QWidget()
        line_layout = QVBoxLayout(self.line_plot_widget)
        line_layout.setContentsMargins(20, 5, 20, 5) # padx, pady
        
        # This is the rounded-corner frame (the "why")
        self.line_plot_frame = QFrame()
        self.line_plot_frame.setStyleSheet("background-color: #F0F0F0; border-radius: 20px;")
        line_layout.addWidget(self.line_plot_frame)
        
        plot_layout = QVBoxLayout(self.line_plot_frame)
        plot_layout.setContentsMargins(15, 15, 15, 15)
        
        self.line_fig, self.line_ax = plt.subplots()
        self.line_canvas = FigureCanvasQTAgg(self.line_fig)
        plot_layout.addWidget(self.line_canvas)
        
        self.plot_stack.addWidget(self.line_plot_widget)
        
        # --- Plot 2: Scatter Plot ---
        self.scatter_plot_widget = QWidget()
        scatter_layout = QVBoxLayout(self.scatter_plot_widget)
        scatter_layout.setContentsMargins(20, 5, 20, 5)
        
        self.scatter_plot_frame = QFrame()
        self.scatter_plot_frame.setStyleSheet("background-color: #F0F0F0; border-radius: 20px;")
        scatter_layout.addWidget(self.scatter_plot_frame)
        
        scatter_plot_layout = QVBoxLayout(self.scatter_plot_frame)
        scatter_plot_layout.setContentsMargins(15, 15, 15, 15)

        self.scatter_fig, self.scatter_ax = plt.subplots()
        self.scatter_canvas = FigureCanvasQTAgg(self.scatter_fig)
        scatter_plot_layout.addWidget(self.scatter_canvas)
        
        self.plot_stack.addWidget(self.scatter_plot_widget)

        # --- Footer Legend Area ---
        footer_frame = QFrame(self)
        footer_frame.setFixedHeight(70)
        footer_layout = QHBoxLayout(footer_frame)
        footer_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- Line Plot Legend (Page 1 of stack) ---
        self.line_legend_stack = QStackedWidget()
        footer_layout.addWidget(self.line_legend_stack)
        
        self.line_legend_frame = QFrame()
        line_legend_layout = QHBoxLayout(self.line_legend_frame)
        line_legend_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.cpu_label_line = self.create_legend_item(line_legend_layout, "#E53935", "CPU : 0%")
        self.ram_label_line = self.create_legend_item(line_legend_layout, "#43A047", "RAM : 0%")
        self.disk_label_line = self.create_legend_item(line_legend_layout, "#1E88E5", "DISK : 0%")
        self.line_legend_stack.addWidget(self.line_legend_frame)

        # --- Scatter Plot Legend (Page 2 of stack) ---
        self.scatter_legend_frame = QFrame()
        scatter_legend_layout = QHBoxLayout(self.scatter_legend_frame)
        scatter_legend_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.cpu_label_scatter = QLabel("CPU : 0%")
        self.ram_label_scatter = QLabel("RAM : 0%")
        self.disk_label_scatter = QLabel("DISK : 0%")
        
        for label in [self.cpu_label_scatter, self.ram_label_scatter, self.disk_label_scatter]:
            label.setFont(self.legend_font)
            label.setStyleSheet("color: #FFFFFF; padding: 0 20px;")
            scatter_legend_layout.addWidget(label)
        self.line_legend_stack.addWidget(self.scatter_legend_frame)

        main_layout.addWidget(footer_frame)

    def create_legend_item(self, layout, color, text):
        """Helper to create the dotted legend items."""
        item_frame = QFrame()
        item_layout = QHBoxLayout(item_frame)
        item_layout.setContentsMargins(0, 0, 0, 0)
        
        color_dot = QFrame()
        color_dot.setFixedSize(15, 15)
        color_dot.setStyleSheet(f"background-color: {color}; border-radius: 7px;")
        
        label = QLabel(text)
        label.setFont(self.legend_font)
        label.setStyleSheet("color: #FFFFFF; padding-left: 10px; padding-right: 20px;")
        
        item_layout.addWidget(color_dot)
        item_layout.addWidget(label)
        layout.addWidget(item_frame)
        return label

    def init_worker_thread(self):
        """Create, connect, and start the background data thread."""
        
        # 1. Create the worker and thread
        self.worker = SystemDataWorker()
        self.thread = QThread()
        
        # 2. Move worker to thread
        self.worker.moveToThread(self.thread)
        
        # ======================================================================
        # STEP 3: CONNECT SIGNALS AND SLOTS
        # ======================================================================
        
        # Connect worker signals to GUI slots
        self.worker.dataUpdated.connect(self.update_line_plot)
        self.worker.processDataUpdated.connect(self.update_scatter_plot)
        
        # Connect thread started signal to worker's start method
        self.thread.started.connect(self.worker.start_monitoring)
        
        # Connect window's close event to stop the thread
        self.destroyed.connect(self.stop_worker)
        
        # 3. Start the thread
        self.thread.start()

    def stop_worker(self):
        """Safely stop the worker and thread."""
        print("Stopping worker...")
        self.worker.stop_monitoring()
        self.thread.quit()
        self.thread.wait() # Wait for thread to finish
        print("Worker stopped.")
        
    def closeEvent(self, event):
        """Override close event to stop the thread."""
        self.stop_worker()
        event.accept()
        

    # ==========================================================================
    # --- GUI UPDATE SLOTS (These receive the data) ---
    # ==========================================================================
    
    @Slot(dict, pd.DataFrame)
    def update_line_plot(self, metrics, data):
        """
        This function is now a SLOT. It only runs when the
        worker sends new data. It ONLY does drawing.
        """
        
        # Update labels in BOTH legends
        for label in [self.cpu_label_line, self.cpu_label_scatter]:
            label.setText(f"CPU : {metrics['cpu']:.0f}%")
        for label in [self.ram_label_line, self.ram_label_scatter]:
            label.setText(f"RAM : {metrics['ram']:.0f}%")
        for label in [self.disk_label_line, self.disk_label_scatter]:
            label.setText(f"DISK : {metrics['disk']:.0f}%")
            
        # --- Update Line Plot (Your logic, moved here) ---
        self.line_ax.clear()
        
        if not data.empty:
            current_time = data['timestamp'].iloc[-1]
            time_window_ago = current_time - self.display_time_window
            recent_data = data[data['timestamp'] >= time_window_ago].copy()
            times = (recent_data['timestamp'] - current_time).values
            
            if not recent_data.empty:
                first_cpu = recent_data['cpu'].iloc[0]
                first_ram = recent_data['ram'].iloc[0]
                first_disk = recent_data['disk'].iloc[0]
                first_time = times[0]
                
                if first_time > -self.display_time_window:
                    baseline_times = [-self.display_time_window, first_time]
                    self.line_ax.plot(baseline_times, [first_cpu, first_cpu], color='#E53935', 
                                    linewidth=1.5, alpha=0.4, linestyle='--', label='CPU Baseline')
                    self.line_ax.plot(baseline_times, [first_ram, first_ram], color='#43A047', 
                                    linewidth=1.5, alpha=0.4, linestyle='--', label='RAM Baseline')
                    self.line_ax.plot(baseline_times, [first_disk, first_disk], color='#1E88E5', 
                                    linewidth=1.5, alpha=0.4, linestyle='--', label='DISK Baseline')
        else:
            recent_data = data
            times = []
            baseline_times = [-self.display_time_window, 0]
            self.line_ax.plot(baseline_times, [0, 0], color='#E53935', linewidth=1, alpha=0.3, linestyle='--')
            self.line_ax.plot(baseline_times, [0, 0], color='#43A047', linewidth=1, alpha=0.3, linestyle='--')
            self.line_ax.plot(baseline_times, [0, 0], color='#1E88E5', linewidth=1, alpha=0.3, linestyle='--')
        
        if len(times) > 5:
            try:
                times_smooth = np.linspace(times.min(), times.max(), len(times) + 10)
                cpu_spline = make_interp_spline(times, recent_data['cpu'], k=2)
                cpu_smooth = cpu_spline(times_smooth)
                self.line_ax.plot(times_smooth, cpu_smooth, color='#E53935', linewidth=2.5, 
                                alpha=0.8, label='CPU')
                ram_spline = make_interp_spline(times, recent_data['ram'], k=2)
                ram_smooth = ram_spline(times_smooth)
                self.line_ax.plot(times_smooth, ram_smooth, color='#43A047', linewidth=2.5, 
                                alpha=0.8, label='RAM')
                disk_spline = make_interp_spline(times, recent_data['disk'], k=2)
                disk_smooth = disk_spline(times_smooth)
                self.line_ax.plot(times_smooth, disk_smooth, color='#1E88E5', linewidth=2.5, 
                                alpha=0.8, label='DISK')
                
                self.line_ax.fill_between(times_smooth, cpu_smooth, alpha=0.1, color='#E53935')
                self.line_ax.fill_between(times_smooth, ram_smooth, alpha=0.1, color='#43A047')
                self.line_ax.fill_between(times_smooth, disk_smooth, alpha=0.1, color='#1E88E5')
                
                self.line_ax.scatter(times, recent_data['cpu'], color='#E53935', s=15, alpha=0.6, zorder=5)
                self.line_ax.scatter(times, recent_data['ram'], color='#43A047', s=15, alpha=0.6, zorder=5)
                self.line_ax.scatter(times, recent_data['disk'], color='#1E88E5', s=15, alpha=0.6, zorder=5)
                
            except Exception:
                self.line_ax.plot(times, recent_data['cpu'], color='#E53935', linewidth=2, alpha=0.8)
                self.line_ax.plot(times, recent_data['ram'], color='#43A047', linewidth=2, alpha=0.8)
                self.line_ax.plot(times, recent_data['disk'], color='#1E88E5', linewidth=2, alpha=0.8)
                self.line_ax.fill_between(times, recent_data['cpu'], alpha=0.1, color='#E53935')
                self.line_ax.fill_between(times, recent_data['ram'], alpha=0.1, color='#43A047')
                self.line_ax.fill_between(times, recent_data['disk'], alpha=0.1, color='#1E88E5')
        elif len(times) > 1:
            self.line_ax.plot(times, recent_data['cpu'], color='#E53935', linewidth=2, 
                            marker='o', markersize=3, alpha=0.8)
            self.line_ax.plot(times, recent_data['ram'], color='#43A047', linewidth=2, 
                            marker='s', markersize=3, alpha=0.8)
            self.line_ax.plot(times, recent_data['disk'], color='#1E88E5', linewidth=2, 
                            marker='^', markersize=3, alpha=0.8)
            self.line_ax.fill_between(times, recent_data['cpu'], alpha=0.1, color='#E53935')
            self.line_ax.fill_between(times, recent_data['ram'], alpha=0.1, color='#43A047')
            self.line_ax.fill_between(times, recent_data['disk'], alpha=0.1, color='#1E88E5')
        elif len(times) == 1:
            self.line_ax.scatter(times, recent_data['cpu'], color='#E53935', s=25, alpha=0.8, zorder=5)
            self.line_ax.scatter(times, recent_data['ram'], color='#43A047', s=25, alpha=0.8, zorder=5)
            self.line_ax.scatter(times, recent_data['disk'], color='#1E88E5', s=25, alpha=0.8, zorder=5)
            
            point_width = self.display_time_window * 0.02
            self.line_ax.fill_between([times[0] - point_width, times[0] + point_width], 
                                    [recent_data['cpu'].iloc[0], recent_data['cpu'].iloc[0]], 
                                    alpha=0.1, color='#E53935')
            self.line_ax.fill_between([times[0] - point_width, times[0] + point_width], 
                                    [recent_data['ram'].iloc[0], recent_data['ram'].iloc[0]], 
                                    alpha=0.1, color='#43A047')
            self.line_ax.fill_between([times[0] - point_width, times[0] + point_width], 
                                    [recent_data['disk'].iloc[0], recent_data['disk'].iloc[0]], 
                                    alpha=0.1, color='#1E88E5')
        
        self.line_ax.set_xlim(-self.display_time_window, 0)
        self.line_ax.set_xlabel("Timeline", color='black', fontsize=12)
        self.line_ax.set_ylabel("Usage (%)", color='black', fontsize=12)
        self.line_ax.set_ylim(0, 100)
        for spine in ['top', 'right']: self.line_ax.spines[spine].set_visible(False)
        self.line_ax.spines['bottom'].set_color('black')
        self.line_ax.spines['left'].set_color('black') # Make left spine visible
        self.line_ax.tick_params(axis='y', colors='black')
        self.line_ax.tick_params(axis='x', colors='black')
        
        # Set all plot text/lines to black
        for text in self.line_ax.get_xticklabels() + self.line_ax.get_yticklabels():
            text.set_color('black')
        self.line_ax.xaxis.label.set_color('black')
        self.line_ax.yaxis.label.set_color('black')
        
        self.line_fig.tight_layout(pad=0.5)
        self.line_canvas.draw()


    @Slot(pd.DataFrame)
    def update_scatter_plot(self, process_df):
        """
        This function is now a SLOT. It only runs when the
        worker sends new process data.
        """
        
        # --- Update Scatter Plot (Your logic, moved here) ---
        self.scatter_ax.clear()
        
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
                x_vals, y_vals, colors, sizes, x_tick_labels, process_names = [], [], [], [], [], []
                user_colors = ['#E53935', '#43A047', '#1E88E5', '#FF9800', '#9C27B0', '#00BCD4', '#795548', '#607D8B']
                background_colors = ['#FFEBEE', '#E8F5E8', '#E3F2FD', '#FFF3E0', '#F3E5F5', '#E0F2F1', '#EFEBE9', '#ECEFF1']

                current_x_offset = 0
                for i, user in enumerate(users):
                    user_procs = filtered_df[filtered_df['username'] == user]
                    user_color = user_colors[i % len(user_colors)]
                    bg_color = background_colors[i % len(background_colors)]
                    user_x_indices = list(range(current_x_offset, current_x_offset + len(user_procs)))
                    
                    x_vals.extend(user_x_indices)
                    y_vals.extend(user_procs['cpu_percent'].tolist())
                    process_names.extend(user_procs['name'].tolist())
                    
                    for cpu_val in user_procs['cpu_percent']:
                        colors.append(user_color)
                        size = max(20, min(100, cpu_val * 3))
                        sizes.append(size)
                    
                    if not user_procs.empty:
                        center_x_pos = current_x_offset + (len(user_procs) - 1) / 2
                        x_tick_labels.append((center_x_pos, user))
                        
                        # Add background rectangles
                        rect = mpatches.Rectangle((current_x_offset - 0.5, 0),
                                            len(user_procs), 100, # Use 100 for height
                                            facecolor=bg_color, edgecolor='none', 
                                            alpha=0.3, zorder=0)
                        self.scatter_ax.add_patch(rect)

                    current_x_offset += len(user_procs) + 1

                self.scatter_ax.scatter(x_vals, y_vals, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=2)
                
                for j, (x, y) in enumerate(zip(x_vals, y_vals)):
                    name = process_names[j][:9] + '...' if len(process_names[j]) > 12 else process_names[j]
                    y_offset = 8 if y > 50 else -15
                    v_align = 'bottom' if y > 50 else 'top'
                    self.scatter_ax.annotate(name, (x, y), textcoords="offset points",
                                            xytext=(0, y_offset), ha='center', va=v_align,
                                            fontsize=6, color='black', alpha=0.8, weight='bold')

                max_cpu = max(y_vals) if y_vals else 0
                self.scatter_ax.set_ylim(0, max(max_cpu * 1.3, 10))
                
                tick_positions = [pos for pos, _ in x_tick_labels]
                tick_labels_text = [label for _, label in x_tick_labels]
                self.scatter_ax.set_xticks(tick_positions)
                self.scatter_ax.set_xticklabels(tick_labels_text, rotation=0, ha="center", fontsize=9, color='black')
                
                self.scatter_ax.tick_params(axis='y', colors='black')
                for spine in ['top', 'right']: self.scatter_ax.spines[spine].set_visible(False)
                self.scatter_ax.spines['bottom'].set_color('black')
                self.scatter_ax.spines['left'].set_color('black')
                self.scatter_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                self.scatter_ax.grid(True, alpha=0.3, linestyle='--', color='gray', zorder=1)
                
            else:
                self.scatter_ax.text(0.5, 0.5, 'No significant processes found', 
                                    transform=self.scatter_ax.transAxes, ha='center', va='center',
                                    fontsize=12, color='gray', alpha=0.7)
        else:
            self.scatter_ax.text(0.5, 0.5, 'Loading process data...', 
                                transform=self.scatter_ax.transAxes, ha='center', va='center',
                                fontsize=12, color='gray', alpha=0.7)
        
        # Set all plot text/lines to black
        for text in self.scatter_ax.get_xticklabels() + self.scatter_ax.get_yticklabels():
            text.set_color('black')
        
        self.scatter_fig.tight_layout(pad=0.5)
        self.scatter_canvas.draw()

    # --- GUI Control Slots ---
    
    @Slot()
    def switch_view(self, index):
        """Switches the main view and legend."""
        self.plot_stack.setCurrentIndex(index)
        self.line_legend_stack.setCurrentIndex(index)
        
        if index == 0:
            # "Usage" view
            self.usage_button.setFont(self.nav_font_bold)
            self.usage_button.setStyleSheet(self.usage_button.styleSheet() + "color: #FFFFFF;")
            self.scatter_button.setFont(self.nav_font_normal)
            self.scatter_button.setStyleSheet(self.scatter_button.styleSheet() + "color: #A0A0A0;")
        else:
            # "Scatter" view
            self.usage_button.setFont(self.nav_font_normal)
            self.usage_button.setStyleSheet(self.usage_button.styleSheet() + "color: #A0A0A0;")
            self.scatter_button.setFont(self.nav_font_bold)
            self.scatter_button.setStyleSheet(self.scatter_button.styleSheet() + "color: #FFFFFF;")


# ==============================================================================
# --- APPLICATION ENTRY POINT ---
# ==============================================================================

if __name__ == "__main__":
    # Handle DPI scaling
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())