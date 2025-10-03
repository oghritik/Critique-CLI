import psutil
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.interpolate import make_interp_spline
import time

# --- Force consistent DPI and scaling ---
import os
import sys

# Force DPI awareness on Windows
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # System DPI aware
    except:
        pass

# Force matplotlib backend and DPI settings
os.environ['MPLBACKEND'] = 'TkAgg'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'

# --- Matplotlib style setup with aggressive DPI control ---
import matplotlib
matplotlib.use('TkAgg', force=True)

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.dpi': 100,  # Restore readable DPI
    'savefig.dpi': 100,
    'figure.figsize': (8, 4),  # Restore original figure size
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
})

class SystemMonitor:
    def __init__(self):
        self.data = pd.DataFrame(columns=['timestamp', 'cpu', 'ram', 'disk'])
        self.monitoring = False
        self.max_history = 60  # Keep last 60 seconds of data
        self.display_time_window = 30  # Display last 30 seconds (configurable)
        self.root = None
        self.active_view = "usage"

    def collect_metrics(self):
        """Collect real-time system metrics."""
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        return {'cpu': cpu, 'ram': ram, 'disk': disk}

    def monitor_system(self):
        """Update dataset with latest metrics."""
        metrics = self.collect_metrics()
        timestamp = time.time()
        new_data = {'timestamp': timestamp, **metrics}
        
        # Fix pandas FutureWarning by ensuring proper DataFrame structure
        if self.data.empty:
            self.data = pd.DataFrame([new_data])
        else:
            new_row = pd.DataFrame([new_data])
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            
        if len(self.data) > self.max_history:
            self.data = self.data.iloc[-self.max_history:]
        return metrics

    # def get_process_data(self):
    #     """Collect CPU usage of all processes grouped by username."""
    #     processes = []
    #     for proc in psutil.process_iter(['name', 'cpu_percent', 'username']):
    #         try:
    #             if proc.info['cpu_percent'] is not None and proc.info['cpu_percent'] > 0.1:
    #                 processes.append(proc.info)
    #         except (psutil.NoSuchProcess, psutil.AccessDenied):
    #             continue
    #     df = pd.DataFrame(processes)
    #     if df.empty:
    #         return pd.DataFrame(), {}
    #     # Group by username and aggregate process data
    #     user_grouped = df.groupby('username')['cpu_percent'].apply(list).to_dict()
    #     return df, user_grouped
    
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

    def switch_view(self, view_name):
        """Switch between the 'usage' and 'scatter' plot views."""
        self.active_view = view_name
        if view_name == "usage":
            # Show usage plot and its legend
            self.line_plot_frame.tkraise()
            self.line_legend_frame.pack(expand=True)
            self.scatter_legend_frame.pack_forget()
            # Update button styles
            self.usage_button.configure(text_color="#FFFFFF", font=self.nav_font_bold)
            self.scatter_button.configure(text_color="#A0A0A0", font=self.nav_font_normal)
        else:
            # Show scatter plot and its legend
            self.scatter_plot_frame.tkraise()
            self.scatter_legend_frame.pack(expand=True)
            self.line_legend_frame.pack_forget()
            # Update button styles
            self.scatter_button.configure(text_color="#FFFFFF", font=self.nav_font_bold)
            self.usage_button.configure(text_color="#A0A0A0", font=self.nav_font_normal)

    def update_font_size(self, event=None):
        """Dynamically adjust font sizes based on window width and DPI."""
        if not self.root: return
        
        try:
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            
            # Calculate scale factor based on window size
            base_width = 800
            base_height = 500
            width_scale = width / base_width
            height_scale = height / base_height
            scale_factor = min(width_scale, height_scale)  # Use smaller scale to prevent oversizing
            
            # Clamp scale factor to reasonable bounds
            scale_factor = max(0.8, min(2.0, scale_factor))
            
            nav_size = max(12, min(24, int(16 * scale_factor)))
            legend_size = max(10, min(20, int(14 * scale_factor)))

            self.nav_font_bold.configure(size=nav_size)
            self.nav_font_normal.configure(size=nav_size)
            self.legend_font.configure(size=legend_size)
            
            # Update matplotlib font sizes
            plt.rcParams.update({
                'font.size': max(8, min(12, int(10 * scale_factor))),
                'axes.labelsize': max(8, min(12, int(10 * scale_factor))),
                'xtick.labelsize': max(7, min(10, int(9 * scale_factor))),
                'ytick.labelsize': max(7, min(10, int(9 * scale_factor))),
            })
        except:
            pass  # Ignore errors during font scaling

    def display_gui(self):
        # --- Aggressive DPI and scaling control ---
        ctk.set_appearance_mode("dark")
        
        # Try to detect and override system scaling
        try:
            import tkinter as tk
            temp_root = tk.Tk()
            temp_root.withdraw()
            
            # Get actual vs reported screen dimensions to detect scaling
            screen_width = temp_root.winfo_screenwidth()
            screen_height = temp_root.winfo_screenheight()
            actual_width = temp_root.winfo_vrootwidth()
            actual_height = temp_root.winfo_vrootheight()
            
            # Calculate scaling factor
            scale_x = actual_width / screen_width if screen_width > 0 else 1.0
            scale_y = actual_height / screen_height if screen_height > 0 else 1.0
            detected_scale = max(scale_x, scale_y)
            
            temp_root.destroy()
            
            # Force scaling based on detection
            if detected_scale > 1.2:  # High DPI detected
                ctk.set_widget_scaling(0.8)  # Reduce scaling
                ctk.set_window_scaling(0.8)
            else:
                ctk.set_widget_scaling(1.0)
                ctk.set_window_scaling(1.0)
                
        except:
            # Fallback to standard scaling
            ctk.set_widget_scaling(1.0)
            ctk.set_window_scaling(1.0)
        
        self.root = ctk.CTk()
        self.root.title("System Monitor")
        
        # --- Original window size with DPI fixes ---
        self.root.geometry("700x450")
        self.root.configure(fg_color="#212121")
        self.root.minsize(700, 450)

        # --- Fonts ---
        self.nav_font_bold = ctk.CTkFont("Arial", 15, "bold")
        self.nav_font_normal = ctk.CTkFont("Arial", 15)
        self.legend_font = ctk.CTkFont("Arial", 16)
        self.root.bind("<Configure>", self.update_font_size)
        
        # --- ESC key binding to exit ---
        def on_escape(event):
            self.monitoring = False
            self.root.quit()
            self.root.destroy()
        
        self.root.bind("<Escape>", on_escape)
        self.root.focus_set()  # Ensure window can receive key events

        # --- Main UI Grid (Title, Nav, Content, Footer) ---
        self.root.grid_rowconfigure(0, weight=0)  # Title
        self.root.grid_rowconfigure(1, weight=0)  # Navigation
        self.root.grid_rowconfigure(2, weight=1)  # Main Content/Plot
        self.root.grid_rowconfigure(3, weight=0)  # Footer/Legend
        self.root.grid_columnconfigure(0, weight=1)

        # --- Custom Navigation Header ---
        nav_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        nav_frame.grid(row=1, column=0, sticky="ew", padx=25, pady=15)
        self.usage_button = ctk.CTkButton(nav_frame, text="SYSTEM USAGE", command=lambda: self.switch_view("usage"), fg_color="transparent", hover=False)
        self.usage_button.pack(side="left", padx=(0, 20))
        self.scatter_button = ctk.CTkButton(nav_frame, text="SCATTER PLOT", command=lambda: self.switch_view("scatter"), fg_color="transparent", hover=False)
        self.scatter_button.pack(side="left")

        # --- Main Content Area (for plots) with curved frame ---
        content_area = ctk.CTkFrame(self.root, fg_color="#F0F0F0", corner_radius=20, border_width=0.5, border_color="#D0D0D0")
        content_area.grid(row=2, column=0, sticky="nsew", padx=20, pady=5)
        content_area.grid_propagate(False)
        content_area.grid_rowconfigure(0, weight=1)
        content_area.grid_columnconfigure(0, weight=1)
        
        # --- Line Plot Frame & Canvas ---
        self.line_plot_frame = ctk.CTkFrame(content_area, fg_color="transparent", corner_radius=20)
        self.line_plot_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create figure with DPI control but readable size
        self.line_fig, self.line_ax = plt.subplots(figsize=(8, 4), dpi=100, facecolor="#F0F0F0")
        self.line_ax.set_facecolor("#F0F0F0")
        self.line_canvas = FigureCanvasTkAgg(self.line_fig, master=self.line_plot_frame)
        self.line_canvas.get_tk_widget().pack(expand=True, fill="both", padx=15, pady=15)

        # --- Scatter Plot Frame & Canvas ---
        self.scatter_plot_frame = ctk.CTkFrame(content_area, fg_color="transparent", corner_radius=20)
        self.scatter_plot_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create figure with DPI control but readable size
        self.scatter_fig, self.scatter_ax = plt.subplots(figsize=(8, 4), dpi=100, facecolor="#F0F0F0")
        self.scatter_ax.set_facecolor("#F0F0F0")
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=self.scatter_plot_frame)
        self.scatter_canvas.get_tk_widget().pack(expand=True, fill="both", padx=15, pady=15)

        # --- Footer Legend Area ---
        footer_frame = ctk.CTkFrame(self.root, fg_color="transparent", height=60)
        footer_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        # --- Line Plot Legend (with color dots) ---
        self.line_legend_frame = ctk.CTkFrame(footer_frame, fg_color="transparent")
        # Items for line legend
        cpu_item_line = ctk.CTkFrame(self.line_legend_frame, fg_color="transparent")
        cpu_item_line.pack(side="left", padx=20)
        ctk.CTkFrame(cpu_item_line, width=15, height=15, fg_color="#E53935", corner_radius=10).pack(side="left")
        self.cpu_label_line = ctk.CTkLabel(cpu_item_line, text="CPU : 0%", font=self.legend_font, text_color="#FFFFFF")
        self.cpu_label_line.pack(side="left", padx=10)
        # ... (similar for RAM and DISK)
        ram_item_line = ctk.CTkFrame(self.line_legend_frame, fg_color="transparent")
        ram_item_line.pack(side="left", padx=20)
        ctk.CTkFrame(ram_item_line, width=15, height=15, fg_color="#43A047", corner_radius=10).pack(side="left")
        self.ram_label_line = ctk.CTkLabel(ram_item_line, text="RAM : 0%", font=self.legend_font, text_color="#FFFFFF")
        self.ram_label_line.pack(side="left", padx=10)
        disk_item_line = ctk.CTkFrame(self.line_legend_frame, fg_color="transparent")
        disk_item_line.pack(side="left", padx=20)
        ctk.CTkFrame(disk_item_line, width=15, height=15, fg_color="#1E88E5", corner_radius=10).pack(side="left")
        self.disk_label_line = ctk.CTkLabel(disk_item_line, text="DISK : 0%", font=self.legend_font, text_color="#FFFFFF")
        self.disk_label_line.pack(side="left", padx=10)

        # --- Scatter Plot Legend (plain text) ---
        self.scatter_legend_frame = ctk.CTkFrame(footer_frame, fg_color="transparent")
        # Items for scatter legend
        self.cpu_label_scatter = ctk.CTkLabel(self.scatter_legend_frame, text="CPU : 0%", font=self.legend_font, text_color="#FFFFFF")
        self.cpu_label_scatter.pack(side="left", padx=20)
        self.ram_label_scatter = ctk.CTkLabel(self.scatter_legend_frame, text="RAM : 0%", font=self.legend_font, text_color="#FFFFFF")
        self.ram_label_scatter.pack(side="left", padx=20)
        self.disk_label_scatter = ctk.CTkLabel(self.scatter_legend_frame, text="DISK : 0%", font=self.legend_font, text_color="#FFFFFF")
        self.disk_label_scatter.pack(side="left", padx=20)

        def update_gui():
            if not self.monitoring: 
                return
            
            try:
                metrics = self.monitor_system()
            except Exception as e:
                # Handle any errors in monitoring and continue
                print(f"Monitoring error: {e}")
                if self.monitoring:
                    self.root.after(500, update_gui)
                return
            # Update labels in BOTH legends
            for label in [self.cpu_label_line, self.cpu_label_scatter]:
                label.configure(text=f"CPU : {metrics['cpu']:.0f}%")
            for label in [self.ram_label_line, self.ram_label_scatter]:
                label.configure(text=f"RAM : {metrics['ram']:.0f}%")
            for label in [self.disk_label_line, self.disk_label_scatter]:
                label.configure(text=f"DISK : {metrics['disk']:.0f}%")

            # --- Update Line Plot with configurable time window ---
            self.line_ax.clear()
            
            # Filter data to last display_time_window seconds only
            if not self.data.empty:
                current_time = self.data['timestamp'].iloc[-1]
                time_window_ago = current_time - self.display_time_window
                recent_data = self.data[self.data['timestamp'] >= time_window_ago].copy()
                times = (recent_data['timestamp'] - current_time).values
                
                # Get first entry values for baseline
                if not recent_data.empty:
                    first_cpu = recent_data['cpu'].iloc[0]
                    first_ram = recent_data['ram'].iloc[0]
                    first_disk = recent_data['disk'].iloc[0]
                    first_time = times[0]
                    
                    # Draw baseline lines from -display_time_window to first data point using first entry values
                    if first_time > -self.display_time_window:
                        baseline_times = [-self.display_time_window, first_time]
                        self.line_ax.plot(baseline_times, [first_cpu, first_cpu], color='#E53935', 
                                        linewidth=1.5, alpha=0.4, linestyle='--', label='CPU Baseline')
                        self.line_ax.plot(baseline_times, [first_ram, first_ram], color='#43A047', 
                                        linewidth=1.5, alpha=0.4, linestyle='--', label='RAM Baseline')
                        self.line_ax.plot(baseline_times, [first_disk, first_disk], color='#1E88E5', 
                                        linewidth=1.5, alpha=0.4, linestyle='--', label='DISK Baseline')
            else:
                recent_data = self.data
                times = []
                # Show zero baseline when no data exists
                baseline_times = [-self.display_time_window, 0]
                self.line_ax.plot(baseline_times, [0, 0], color='#E53935', linewidth=1, alpha=0.3, linestyle='--')
                self.line_ax.plot(baseline_times, [0, 0], color='#43A047', linewidth=1, alpha=0.3, linestyle='--')
                self.line_ax.plot(baseline_times, [0, 0], color='#1E88E5', linewidth=1, alpha=0.3, linestyle='--')
            
            # Moderate smooth curves with reduced interpolation
            if len(times) > 5:  # Need at least 6 points for gentle splines
                try:
                    # Create gentler spline curves with less interpolation
                    times_smooth = np.linspace(times.min(), times.max(), len(times) + 10)  # Reduced from * 3
                    
                    # CPU curve with gentler spline interpolation
                    cpu_spline = make_interp_spline(times, recent_data['cpu'], k=2)  # Reduced from k=3
                    cpu_smooth = cpu_spline(times_smooth)
                    self.line_ax.plot(times_smooth, cpu_smooth, color='#E53935', linewidth=2.5, 
                                    alpha=0.8, label='CPU')
                    
                    # RAM curve with gentler spline interpolation  
                    ram_spline = make_interp_spline(times, recent_data['ram'], k=2)  # Reduced from k=3
                    ram_smooth = ram_spline(times_smooth)
                    self.line_ax.plot(times_smooth, ram_smooth, color='#43A047', linewidth=2.5, 
                                    alpha=0.8, label='RAM')
                    
                    # DISK curve with gentler spline interpolation
                    disk_spline = make_interp_spline(times, recent_data['disk'], k=2)  # Reduced from k=3
                    disk_smooth = disk_spline(times_smooth)
                    self.line_ax.plot(times_smooth, disk_smooth, color='#1E88E5', linewidth=2.5, 
                                    alpha=0.8, label='DISK')
                    
                    # Add lighter fill areas under curves - start from first reading
                    self.line_ax.fill_between(times_smooth, cpu_smooth, alpha=0.1, color='#E53935')
                    self.line_ax.fill_between(times_smooth, ram_smooth, alpha=0.1, color='#43A047')
                    self.line_ax.fill_between(times_smooth, disk_smooth, alpha=0.1, color='#1E88E5')
                    
                    # Add smaller data points as markers
                    self.line_ax.scatter(times, recent_data['cpu'], color='#E53935', s=15, alpha=0.6, zorder=5)
                    self.line_ax.scatter(times, recent_data['ram'], color='#43A047', s=15, alpha=0.6, zorder=5)
                    self.line_ax.scatter(times, recent_data['disk'], color='#1E88E5', s=15, alpha=0.6, zorder=5)
                    
                except Exception:
                    # Fallback to regular lines if spline fails
                    self.line_ax.plot(times, recent_data['cpu'], color='#E53935', linewidth=2, alpha=0.8)
                    self.line_ax.plot(times, recent_data['ram'], color='#43A047', linewidth=2, alpha=0.8)
                    self.line_ax.plot(times, recent_data['disk'], color='#1E88E5', linewidth=2, alpha=0.8)
                    
                    # Add fill areas for fallback lines too
                    self.line_ax.fill_between(times, recent_data['cpu'], alpha=0.1, color='#E53935')
                    self.line_ax.fill_between(times, recent_data['ram'], alpha=0.1, color='#43A047')
                    self.line_ax.fill_between(times, recent_data['disk'], alpha=0.1, color='#1E88E5')
            elif len(times) > 1:
                # Simple lines for fewer data points
                self.line_ax.plot(times, recent_data['cpu'], color='#E53935', linewidth=2, 
                                marker='o', markersize=3, alpha=0.8)
                self.line_ax.plot(times, recent_data['ram'], color='#43A047', linewidth=2, 
                                marker='s', markersize=3, alpha=0.8)
                self.line_ax.plot(times, recent_data['disk'], color='#1E88E5', linewidth=2, 
                                marker='^', markersize=3, alpha=0.8)
                
                # Add fill areas even for simple lines
                self.line_ax.fill_between(times, recent_data['cpu'], alpha=0.1, color='#E53935')
                self.line_ax.fill_between(times, recent_data['ram'], alpha=0.1, color='#43A047')
                self.line_ax.fill_between(times, recent_data['disk'], alpha=0.1, color='#1E88E5')
            elif len(times) == 1:
                # Single data point - show as dot with small fill area
                self.line_ax.scatter(times, recent_data['cpu'], color='#E53935', s=25, alpha=0.8, zorder=5)
                self.line_ax.scatter(times, recent_data['ram'], color='#43A047', s=25, alpha=0.8, zorder=5)
                self.line_ax.scatter(times, recent_data['disk'], color='#1E88E5', s=25, alpha=0.8, zorder=5)
                
                # Add small fill areas around single points
                point_width = self.display_time_window * 0.02  # 2% of time window
                self.line_ax.fill_between([times[0] - point_width, times[0] + point_width], 
                                        [recent_data['cpu'].iloc[0], recent_data['cpu'].iloc[0]], 
                                        alpha=0.1, color='#E53935')
                self.line_ax.fill_between([times[0] - point_width, times[0] + point_width], 
                                        [recent_data['ram'].iloc[0], recent_data['ram'].iloc[0]], 
                                        alpha=0.1, color='#43A047')
                self.line_ax.fill_between([times[0] - point_width, times[0] + point_width], 
                                        [recent_data['disk'].iloc[0], recent_data['disk'].iloc[0]], 
                                        alpha=0.1, color='#1E88E5')
            
            # Set x-axis to show last display_time_window seconds
            self.line_ax.set_xlim(-self.display_time_window, 0)
            
            self.line_ax.set_xlabel("Timeline", color='black', fontsize=12)
            self.line_ax.set_ylabel("Usage (%)", color='black', fontsize=12)
            self.line_ax.set_ylim(0, 100)
            for spine in ['top', 'right']: self.line_ax.spines[spine].set_visible(False)
            self.line_ax.spines['bottom'].set_color('black')
            # self.line_ax.set_yticklabels([]); self.line_ax.set_xticklabels([])
            # self.line_ax.tick_params(axis='y', length=5, color='black'); self.line_ax.tick_params(axis='x', length=0)
            self.line_ax.tick_params(axis='y', colors='black')
            self.line_ax.tick_params(axis='x', colors='black')
            self.line_fig.tight_layout(pad=0.5)
            self.line_canvas.draw()

            # --- Update Enhanced Scatter Plot ---
            self.scatter_ax.clear()
            process_df = self.get_process_data()
            
            if not process_df.empty:
                # Filter processes with meaningful CPU usage
                user_cpu_summary = process_df.groupby('username')['cpu_percent'].sum().reset_index()
                user_threshold = 1.0  # Lowered threshold to show more users (0.5)
                relevant_users = user_cpu_summary[user_cpu_summary['cpu_percent'] > user_threshold]['username'].tolist()

                filtered_df = process_df[process_df['username'].isin(relevant_users)].copy()
                
                process_threshold = 0.5  # Lowered to show more processes (0.1)
                filtered_df = filtered_df[filtered_df['cpu_percent'] >= process_threshold]

                if not filtered_df.empty:
                    filtered_df = filtered_df.sort_values(by=['username', 'cpu_percent'], ascending=[True, False]).reset_index(drop=True)

                    users = filtered_df['username'].unique()
                    x_vals = []
                    y_vals = []
                    colors = []
                    sizes = []
                    x_tick_labels = []
                    process_names = []
                    
                    # Enhanced color palette for different users
                    user_colors = ['#E53935', '#43A047', '#1E88E5', '#FF9800', '#9C27B0', '#00BCD4', '#795548', '#607D8B']
                    background_colors = ['#FFEBEE', '#E8F5E8', '#E3F2FD', '#FFF3E0', '#F3E5F5', '#E0F2F1', '#EFEBE9', '#ECEFF1']

                    # Clear existing patches
                    for rect in self.scatter_ax.patches:
                        rect.remove()

                    current_x_offset = 0
                    for i, user in enumerate(users):
                        user_procs = filtered_df[filtered_df['username'] == user]
                        user_color = user_colors[i % len(user_colors)]
                        bg_color = background_colors[i % len(background_colors)]
                        
                        user_x_indices = list(range(current_x_offset, current_x_offset + len(user_procs)))
                        
                        x_vals.extend(user_x_indices)
                        y_vals.extend(user_procs['cpu_percent'].tolist())
                        process_names.extend(user_procs['name'].tolist())
                        
                        # Color and size based on CPU usage
                        for cpu_val in user_procs['cpu_percent']:
                            colors.append(user_color)
                            # Size based on CPU usage (20-100 range)
                            size = max(20, min(100, cpu_val * 3))
                            sizes.append(size)
                        
                        if not user_procs.empty:
                            center_x_pos = current_x_offset + (len(user_procs) - 1) / 2
                            x_tick_labels.append((center_x_pos, user))

                            # Add background rectangles with rounded appearance
                            rect = plt.Rectangle((current_x_offset - 0.5, self.scatter_ax.get_ylim()[0] if self.scatter_ax.get_ylim()[0] != 0 else 0),
                                               len(user_procs) + 0.5,
                                               100,  # Fixed height for background
                                               facecolor=bg_color, edgecolor='none', alpha=0.3, zorder=0)
                            self.scatter_ax.add_patch(rect)

                        current_x_offset += len(user_procs) + 1

                    # Enhanced scatter plot with variable colors and sizes
                    scatter = self.scatter_ax.scatter(x_vals, y_vals, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=0.5)
                    
                    # Add process name annotations with better positioning
                    for j, (x, y) in enumerate(zip(x_vals, y_vals)):
                        name = process_names[j]
                        if len(name) > 12:
                            name = name[:9] + '...'
                        
                        # Position text above higher CPU usage points, below lower ones
                        y_offset = 8 if y > 50 else -15
                        v_align = 'bottom' if y > 50 else 'top'
                        
                        self.scatter_ax.annotate(name, 
                                               (x, y), 
                                               textcoords="offset points",
                                               xytext=(0, y_offset),
                                               ha='center',
                                               va=v_align,
                                               fontsize=6,
                                               color='black',
                                               alpha=0.8,
                                               weight='bold')

                    # Dynamic y-axis scaling
                    max_cpu = max(y_vals) if y_vals else 0
                    self.scatter_ax.set_ylim(0, max(max_cpu * 1.3, 10))
                    
                    # Set x-axis labels
                    tick_positions = [pos for pos, _ in x_tick_labels]
                    tick_labels_text = [label for _, label in x_tick_labels]
                    
                    self.scatter_ax.set_xticks(tick_positions)
                    self.scatter_ax.set_xticklabels(tick_labels_text, rotation=0, ha="center", fontsize=9, color='black')
                    

                    # self.scatter_ax.set_xlabel("Users", color='black', fontsize=10)
                    # self.scatter_ax.set_ylabel("CPU Usage (%)", color='black', fontsize=10)
                    self.scatter_ax.tick_params(axis='y', colors='black')
                    self.scatter_ax.tick_params(axis='x', colors='black')
                    
                    # Remove top and right spines (borders)
                    for spine in ['top', 'right']: 
                        self.scatter_ax.spines[spine].set_visible(False)
                    self.scatter_ax.spines['bottom'].set_color('black')
                    self.scatter_ax.spines['left'].set_color('black')
                    
                    # Format y-axis labels to show float values
                    from matplotlib.ticker import FormatStrFormatter
                    self.scatter_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    
                    # Add grid for better readability
                    self.scatter_ax.grid(True, alpha=0.3, linestyle='--', color='gray')
                    
                else:
                    # Show message when no processes meet threshold
                    self.scatter_ax.text(0.5, 0.5, 'No significant processes found', 
                                       transform=self.scatter_ax.transAxes, ha='center', va='center',
                                       fontsize=12, color='gray', alpha=0.7)
            else:
                # Show message when no process data available
                self.scatter_ax.text(0.5, 0.5, 'Loading process data...', 
                                   transform=self.scatter_ax.transAxes, ha='center', va='center',
                                   fontsize=12, color='gray', alpha=0.7)
            
            self.scatter_fig.tight_layout(pad=0.5)
            self.scatter_canvas.draw()

            # Safe scheduling of next update
            if self.monitoring and self.root and self.root.winfo_exists():
                try:
                    self.root.after(500, update_gui)
                except Exception:
                    pass  # Ignore tkinter errors during shutdown

        # Initialize the view
        self.monitoring = True
        self.switch_view("usage")
        self.root.after(100, self.update_font_size)
        update_gui()
        self.root.mainloop()

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.display_gui()