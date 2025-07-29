import pandas as pd
import re
from sklearn.ensemble import IsolationForest
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LogAnalyzer:
    def __init__(self):
        self.model = IsolationForest(contamination=0.5, random_state=42)
        self.event_types = ["ERROR", "WARNING", "INFO"]

    def parse_log(self, filename):
        try:
            # Try UTF-8 with BOM, then fall back to UTF-16
            encodings = ['utf-8-sig', 'utf-16-le', 'utf-16-be']
            lines = None
            for encoding in encodings:
                try:
                    with open(filename, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            if lines is None:
                raise UnicodeDecodeError("Failed to decode file with supported encodings", b"", 0, 0, "")
            
            timestamps = []
            messages = []
            for line in lines:
                match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*)', line)
                if match:
                    timestamps.append(match.group(1))
                    messages.append(match.group(2).strip().replace('\ufeff', ''))  # Strip BOM
                else:
                    timestamps.append("")
                    messages.append(line.strip().replace('\ufeff', ''))  # Strip BOM
            
            df = pd.DataFrame({'timestamp': timestamps, 'message': messages})
            df['event_type'] = df['message'].apply(self._get_event_type)
            df['message_length'] = df['message'].apply(len)
            df['is_error'] = df['event_type'] == 'ERROR'
            df['is_warning'] = df['event_type'] == 'WARNING'
            
            return df
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error parsing log file: {e}")
            return None

    def _get_event_type(self, message):
        for event in self.event_types:
            if event in message.upper():
                return event
        return "INFO"

    def analyze_log(self, filename):
        df = self.parse_log(filename)
        if df is None:
            return f"Error: Could not process {filename}\n", None, None
        
        features = df[['message_length', 'is_error', 'is_warning']].values
        if len(features) == 0:
            return "Error: No valid log entries found\n", None, None
        if len(features) < 10:  # Skip anomaly detection for small datasets
            df['is_anomaly'] = 0
        else:
            df['is_anomaly'] = self.model.fit_predict(features)
        anomalies = df[df['is_anomaly'] == -1]
        
        summary = {
            'total_entries': len(df),
            'errors': sum(df['is_error']),
            'warnings': sum(df['is_warning']),
            'anomalies': len(anomalies)
        }
        
        output = f"Log Analysis Summary for {filename}:\n"
        output += f"Total Entries: {summary['total_entries']}\n"
        output += f"Errors: {summary['errors']}\n"
        output += f"Warnings: {summary['warnings']}\n"
        output += f"Anomalies Detected: {summary['anomalies']}\n"
        if summary['anomalies'] > 0:
            output += "\nAnomalous Entries:\n"
            for idx, row in anomalies.iterrows():
                output += f"Line {idx + 1}: {row['message']}\n"
        
        return output, summary, df

    def display_gui(self, filename, summary, df):
        root = ctk.CTk()
        root.title(f"Log Analysis: {filename}")
        
        summary_frame = ctk.CTkFrame(root)
        summary_frame.pack(pady=10, padx=10, fill="x")
        
        ctk.CTkLabel(summary_frame, text=f"Total Entries: {summary['total_entries']}").pack(anchor="w")
        ctk.CTkLabel(summary_frame, text=f"Errors: {summary['errors']}").pack(anchor="w")
        ctk.CTkLabel(summary_frame, text=f"Warnings: {summary['warnings']}").pack(anchor="w")
        ctk.CTkLabel(summary_frame, text=f"Anomalies: {summary['anomalies']}").pack(anchor="w")
        
        if summary['anomalies'] > 0:
            anomaly_frame = ctk.CTkFrame(root)
            anomaly_frame.pack(pady=10, padx=10, fill="both", expand=True)
            anomaly_text = ctk.CTkTextbox(anomaly_frame, height=100)
            anomaly_text.pack(fill="both", expand=True)
            for idx, row in df[df['is_anomaly'] == -1].iterrows():
                anomaly_text.insert("end", f"Line {idx + 1}: {row['message']}\n")
            anomaly_text.configure(state="disabled")

        fig, ax = plt.subplots()
        event_counts = [summary['errors'], summary['warnings'], summary['total_entries'] - summary['errors'] - summary['warnings']]
        ax.bar(['Errors', 'Warnings', 'Info'], event_counts, color=['#ff4d4d', '#ffcc00', '#3399ff'])
        ax.set_title("Log Event Distribution")
        ax.set_ylabel("Count")
        
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, padx=10, fill="both", expand=True)
        
        root.mainloop()