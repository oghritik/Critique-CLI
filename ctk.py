import customtkinter as ctk

ctk.set_appearance_mode("dark")  # or "light"
ctk.set_default_color_theme("blue")  # also "green", "dark-blue", etc.

app = ctk.CTk()  # instead of tk.Tk()
app.geometry("400x300")

button = ctk.CTkButton(master=app, text="Click Me")
button.pack(pady=20)

app.mainloop()