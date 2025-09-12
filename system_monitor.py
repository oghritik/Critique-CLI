import psutil
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# --- Matplotlib style setup for a modern look ---
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'lines.linewidth': 2.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})

class SystemMonitor:
    def __init__(self):
        self.data = pd.DataFrame(columns=['timestamp', 'cpu', 'ram', 'disk'])
        self.monitoring = False
        self.max_history = 60  # Keep last 60 seconds
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
        self.data = pd.concat([self.data, pd.DataFrame([new_data])], ignore_index=True)
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
        """Dynamically adjust font sizes based on window width."""
        if not self.root: return
        width = self.root.winfo_width()
        
        base_width = 1000
        scale_factor = width / base_width

        nav_size = max(18, min(28, int(22 * scale_factor)))
        legend_size = max(14, min(22, int(16 * scale_factor)))

        self.nav_font_bold.configure(size=nav_size)
        self.nav_font_normal.configure(size=nav_size)
        self.legend_font.configure(size=legend_size)

    def display_gui(self):
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("System Monitor")
        self.root.geometry("700x450")
        self.root.configure(fg_color="#212121")

        # --- Fonts ---
        self.nav_font_bold = ctk.CTkFont("Arial", 15, "bold")
        self.nav_font_normal = ctk.CTkFont("Arial", 15)
        self.legend_font = ctk.CTkFont("Arial", 16)
        self.root.bind("<Configure>", self.update_font_size)

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

        # --- Main Content Area (for plots) ---
        content_area = ctk.CTkFrame(self.root, fg_color="#F0F0F0", corner_radius=15)
        content_area.grid(row=2, column=0, sticky="nsew", padx=20, pady=5)
        content_area.grid_propagate(False)
        content_area.grid_rowconfigure(0, weight=1)
        content_area.grid_columnconfigure(0, weight=1)
        
        # --- Line Plot Frame & Canvas ---
        self.line_plot_frame = ctk.CTkFrame(content_area, fg_color="transparent",corner_radius=15)
        self.line_plot_frame.grid(row=0, column=0, sticky="nsew")
        self.line_fig, self.line_ax = plt.subplots(facecolor="#F0F0F0")
        self.line_ax.set_facecolor("#F0F0F0")
        self.line_canvas = FigureCanvasTkAgg(self.line_fig, master=self.line_plot_frame)
        self.line_canvas.get_tk_widget().pack(expand=False , fill="both", padx=10, pady=15)

        # --- Scatter Plot Frame & Canvas ---
        self.scatter_plot_frame = ctk.CTkFrame(content_area, fg_color="transparent",corner_radius=15)
        self.scatter_plot_frame.grid(row=0, column=0, sticky="nsew")
        self.scatter_fig, self.scatter_ax = plt.subplots(facecolor="#F0F0F0")
        self.scatter_ax.set_facecolor("#F0F0F0")
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=self.scatter_plot_frame)
        self.scatter_canvas.get_tk_widget().pack(expand=False, fill="both", padx=5, pady=5)

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
            if not self.monitoring: return
            metrics = self.monitor_system()
            # Update labels in BOTH legends
            for label in [self.cpu_label_line, self.cpu_label_scatter]:
                label.configure(text=f"CPU : {metrics['cpu']:.0f}%")
            for label in [self.ram_label_line, self.ram_label_scatter]:
                label.configure(text=f"RAM : {metrics['ram']:.0f}%")
            for label in [self.disk_label_line, self.disk_label_scatter]:
                label.configure(text=f"DISK : {metrics['disk']:.0f}%")

            # --- Update Line Plot ---
            self.line_ax.clear()
            times = (self.data['timestamp'] - self.data['timestamp'].iloc[-1]).values if not self.data.empty else []            
            self.line_ax.plot(times, self.data['cpu'], color='#E53935', linewidth=1)
            self.line_ax.plot(times, self.data['ram'], color='#43A047', linewidth=1)
            self.line_ax.plot(times, self.data['disk'], color='#1E88E5', linewidth=1)
            
            self.line_ax.set_xlabel("Timeline", color='black', fontsize=12)
            self.line_ax.set_ylabel("Usage (%)", color='black', fontsize=12)
            self.line_ax.set_ylim(0, 100)
            for spine in ['top', 'right']: self.line_ax.spines[spine].set_visible(False)
            self.line_ax.spines['bottom'].set_color('black')
            # self.line_ax.set_yticklabels([]); self.line_ax.set_xticklabels([])
            # self.line_ax.tick_params(axis='y', length=5, color='black'); self.line_ax.tick_params(axis='x', length=0)
            self.line_ax.tick_params(axis='y', colors='black')
            self.line_ax.tick_params(axis='x', colors='black')
            self.line_fig.tight_layout(pad=0.75)
            self.line_canvas.draw()

            # --- Update Scatter Plot ---
            self.scatter_ax.clear()
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

                    for rect in self.scatter_ax.patches:
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

                    self.scatter_ax.scatter(x_vals, y_vals, c='blue', alpha=0.7)
                    
                    # --- NEW: Add process name annotations ---
                    for j, (x, y) in enumerate(zip(x_vals, y_vals)):
                        # Shorten process name if too long, or use '...'
                        name = process_names[j]
                        if len(name) > 15: # Arbitrary length to keep labels concise
                            name = name[:12] + '...'
                        self.scatter_ax.annotate(name, 
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
                    self.scatter_ax.set_ylim(0, max_cpu * 1.2 if max_cpu > 0 else 100)
                    
                    current_x_patch_start = 0
                    for i, user in enumerate(users):
                        user_procs_count = len(filtered_df[filtered_df['username'] == user])
                        
                        rect_color = blue_shade_1 if i % 2 == 0 else blue_shade_2
                        
                        rect = plt.Rectangle((current_x_patch_start - 0.5, self.scatter_ax.get_ylim()[0]),
                                            user_procs_count + 1,
                                            self.scatter_ax.get_ylim()[1] - self.scatter_ax.get_ylim()[0],
                                            facecolor=rect_color, edgecolor='none', zorder=0)
                        self.scatter_ax.add_patch(rect)
                        current_x_patch_start += user_procs_count + 1

                    tick_positions = [pos for pos, _ in x_tick_labels]
                    tick_labels_text = [label for _, label in x_tick_labels]
                    
                    self.scatter_ax.set_xticks(tick_positions)
                    self.scatter_ax.set_xticklabels(tick_labels_text, rotation=0, ha="right", fontsize=8,color='black')
                    
                    self.scatter_ax.set_title("",color='black')
                    self.scatter_ax.set_xlabel("User / Process Category",color='black')
                    self.scatter_ax.set_ylabel("CPU Usage (%)",color='black')
                    self.scatter_ax.tick_params(axis='y',color='black')
            
            self.scatter_fig.tight_layout(pad=3.0)
            self.scatter_canvas.draw()

            self.root.after(500, update_gui)

        # Initialize the view
        self.monitoring = True
        self.switch_view("usage")
        self.root.after(100, self.update_font_size)
        update_gui()
        self.root.mainloop()

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.display_gui()