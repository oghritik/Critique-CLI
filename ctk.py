import customtkinter as ctk
import tkinter as tk
import time
import math

# --- Update function for the clock ---
def update_clock():
    canvas.delete("hands")  # remove old hands
    
    # Get current time
    now = time.localtime()
    hours = now.tm_hour % 12
    minutes = now.tm_min
    seconds = now.tm_sec
    
    # Clock center
    cx, cy = 150, 150
    r = 100  # radius
    
    # Angles in radians
    sec_angle = math.radians(seconds * 6 - 90)
    min_angle = math.radians(minutes * 6 - 90)
    hour_angle = math.radians((hours * 30) + (minutes * 0.5) - 90)
    
    # Second hand
    x_sec = cx + r * 0.9 * math.cos(sec_angle)
    y_sec = cy + r * 0.9 * math.sin(sec_angle)
    canvas.create_line(cx, cy, x_sec, y_sec, fill="red", width=1.5, tags="hands")
    
    # Minute hand
    x_min = cx + r * 0.75 * math.cos(min_angle)
    y_min = cy + r * 0.75 * math.sin(min_angle)
    canvas.create_line(cx, cy, x_min, y_min, fill="black", width=3, tags="hands")
    
    # Hour hand
    x_hour = cx + r * 0.5 * math.cos(hour_angle)
    y_hour = cy + r * 0.5 * math.sin(hour_angle)
    canvas.create_line(cx, cy, x_hour, y_hour, fill="black", width=4, tags="hands")
    
    # Redraw every 1000 ms
    canvas.after(1000, update_clock)


# --- Main Window ---
ctk.set_appearance_mode("light")
root = ctk.CTk()
root.title("Analog Clock in Curved Frame")
root.geometry("400x400")

# Configure grid
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# --- Content Frame with curved edges ---
content_frame = ctk.CTkFrame(
    root,
    fg_color="#F0F0F0",
    corner_radius=50,
    border_width=2,
    border_color="#A0A0A0"
)
content_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

# --- Canvas inside frame for the clock ---
canvas = tk.Canvas(content_frame, width=300, height=300, bg="#F0F0F0", highlightthickness=0)
canvas.pack(padx=20, pady=20)

# Draw clock face
cx, cy = 150, 150
r = 100
canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="black", width=2)

# Hour marks
for i in range(12):
    angle = math.radians(i*30 - 90)
    x1 = cx + r*0.85*math.cos(angle)
    y1 = cy + r*0.85*math.sin(angle)
    x2 = cx + r*0.95*math.cos(angle)
    y2 = cy + r*0.95*math.sin(angle)
    canvas.create_line(x1, y1, x2, y2, fill="black", width=2)

# Start updating hands
update_clock()

root.mainloop()