# ------------------------------------Progress bar-------------------------------------
# from rich.progress import track
# import time
# for step in track(range(10), description="Processing..."):
#     time.sleep(0.2)

# import curses, time

# def bounce(stdscr):
#     stdscr.clear()
#     msg = "Hello!"
#     x = y = 0
#     dx = dy = 1
#     while time.time() < time.time() + 1:  # Run for 10 seconds
#         stdscr.clear()
#         stdscr.addstr(y, x, msg)
#         stdscr.refresh()
#         y, x = y + dy, x + dx
#         if y == 0 or y == curses.LINES - 1:
#             dy *= -1
#         if x == 0 or x == curses.COLS - len(msg):
#             dx *= -1
#         time.sleep(0.05)

# curses.wrapper(bounce)

# from rich.console import Console
# import time

# console = Console()
# with console.status("[bold green]Processing..."):
#     time.sleep(5)  # Simulated work

# console.print("[bold cyan]All done![/bold cyan]")

# ------------------------------------Loading-------------------------------------
# import itertools
# import time
# import sys

# spinner = itertools.cycle(['.', '..', '...', '....'])
# end_time = time.time() + 5  # run for 5 seconds

# while time.time() < end_time:
#     sys.stdout.write('\rLoading' + next(spinner))
#     sys.stdout.flush()
#     time.sleep(0.5)

# print('\rDone!        ')  # clear the line

#--------------------------------------------------------------I am CLI--------------------------------------------------------------   
# from rich.console import Console
# from rich.panel import Panel
# from rich.text import Text

# console = Console()

# # ASCII Art representing "I AM CLI" (edit as you like for shape/size)
# ascii_text1 = """
# ██╗  ░█████╗░███╗░░░███╗  ░█████╗░██╗     ██╗
# ██║  ██╔══██╗████╗░████║  ██╔══██╗██║     ██║
# ██║  ███████║██╔████╔██║  ██║░░╚═╝██║░░░░░██║
# ██║  ██╔══██║██║╚██╔╝██║  ██║░░██╗██║░░░░░██║
# ██║  ██║░░██║██║░╚═╝░██║  ╚█████╔╝███████╗██║
# ╚═╝  ╚═╝░░╚═╝╚═╝░░░░░╚═╝  ░╚════╝░╚══════╝╚═╝
# """
# ascii_text = """
# ██╗   █████╗  ███╗   ███╗   █████╗  ██╗      ██╗
# ██║  ██╔══██╗ ████╗ ████║  ██╔══██╗ ██║      ██║
# ██║  ███████║ ██╔████╔██║  ██║  ╚═╝ ██║      ██║
# ██║  ██╔══██║ ██║╚██╔╝██║  ██║  ██╗ ██║      ██║
# ██║  ██║  ██║ ██║ ╚═╝ ██║  ╚█████╔╝ ███████╗ ██║
# ╚═╝  ╚═╝  ╚═╝ ╚═╝     ╚═╝   ╚════╝  ╚══════╝ ╚═╝
# """

# ascii_text2 = """

# ░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓██████████████▓▒░     ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
# ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
# ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
# ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
# ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
# ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
# ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░ 
                                                                                       
                                                                                       
# """

# # Define gradient colors (start: blue, end: pink)
# start_color = (173, 216, 230)         # lightBlue
# end_color = (255, 105, 180)       # Pink

# def interpolate_color(start, end, t):
#     return tuple(
#         int(start[i] + (end[i] - start[i]) * t) for i in range(3)
#     )

# # Build the banner with per-character gradient
# lines = ascii_text.splitlines()
# banner_text = Text()
# for line in lines:
#     for i, char in enumerate(line):
#         total_chars = len(line)
#         t = i / max(total_chars - 1, 1)
#         r, g, b = interpolate_color(start_color, end_color, t)
#         color_hex = f"#{r:02x}{g:02x}{b:02x}"
#         banner_text.append(char, style=color_hex)
#     banner_text.append("\n")
    
# console.print(banner_text)

# Print with a panel border
# console.print(Panel.fit(banner_text, border_style="cyan", padding=(1, 4)))

# console.print("\n[bold green]Welcome to My Custom CLI!\n")


# mybanner.py
from rich.console import Console
from rich.text import Text

class Animations:
    def __init__(self):
        """Initialize the animation settings."""
        self.console = Console()

        # Default ASCII art for the banner
        self.ascii_text = """
██╗   █████╗  ███╗   ███╗   █████╗  ██╗      ██╗
██║  ██╔══██╗ ████╗ ████║  ██╔══██╗ ██║      ██║
██║  ███████║ ██╔████╔██║  ██║  ╚═╝ ██║      ██║
██║  ██╔══██║ ██║╚██╔╝██║  ██║  ██╗ ██║      ██║
██║  ██║  ██║ ██║ ╚═╝ ██║  ╚█████╔╝ ███████╗ ██║
╚═╝  ╚═╝  ╚═╝ ╚═╝     ╚═╝   ╚════╝  ╚══════╝ ╚═╝
"""
        self.ascii_text2 = """

░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓██████████████▓▒░     ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░ 
                                                                                                                                                      
"""

        # Gradient colors (start: blue, end: pink)
        self.start_color = (173, 216, 230)  # Light blue
        self.end_color = (255, 105, 180)    # Pink

    def banner(self):
        """Print the gradient ASCII banner."""
        
        def interpolate_color(start, end, t):
            """Linearly interpolate between two RGB colors."""
            return tuple(
                int(start[i] + (end[i] - start[i]) * t) for i in range(3)
            )

        lines = self.ascii_text.splitlines()
        banner_text = Text()

        for line in lines:
            for i, char in enumerate(line):
                total_chars = len(line)
                t = i / max(total_chars - 1, 1)
                r, g, b = interpolate_color(self.start_color, self.end_color, t)
                color_hex = f"#{r:02x}{g:02x}{b:02x}"
                banner_text.append(char, style=color_hex)
            banner_text.append("\n")

        self.console.print(banner_text)


# -------- Example usage --------
if __name__ == "__main__":
    anim = Animations()
    anim.banner()
    print("Welcome to My CLI!")