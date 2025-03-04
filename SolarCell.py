# Import library
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataLoader:
     
    def __init__(self, url_2, url_1):
        self.url_1    = url_1
        self.url_2    = url_2
        self.df       = self.load_data()
        self.diff_val = self.load_diff_val()

    def load_data(self):
        df               = pd.read_csv(self.url_2, encoding="utf-8")
        df.columns       = df.columns.str.strip()                                # Clean column names
        df["Time"]       = pd.to_datetime(df["Time"], format="%H:%M")            # Convert time to datetime
        df["Time_float"] = df["Time"].dt.hour + df["Time"].dt.minute / 60        # Convert time to float for plotting
        return df
    
    def load_diff_val(self):
        df_1 = pd.read_csv(self.url_1, encoding="utf-8")
        dust_diff_map = {}

        for _, row in df_1.iterrows():
            dust_level = row["level of dust"]
            dust_diff_map[dust_level] = {
                'V': row["%Diff_V"],
                'I': row["%Diff_I"],
                'P': row["%Diff_P"]
            }
        return dust_diff_map

class DataAnalyzer:
    
    def __init__(self, df, diff_val):
        self.df = df
        self.df_grouped = None
        self.diff_val = diff_val

    def calculate_average(self):
        self.df_grouped = self.df.groupby('Time').agg(
            Power_avg   = ('Power(W)'  , 'mean'),
            Voltage_avg = ('Voltage(V)', 'mean'),
            Current_avg = ('Current(I)', 'mean')
        ).reset_index()
        
        self.df_grouped["Time_float"] = self.df_grouped["Time"].dt.hour + self.df_grouped["Time"].dt.minute / 60

    def calculate_difference(self):
        for dust_level, diff in self.diff_val.items():
            d_V, d_I, d_P = diff["V"], diff["I"], diff["P"]
            
            self.df_grouped[f"Voltage_{dust_level}"] = self.df_grouped["Voltage_avg"] * (1 - d_V / 100)
            self.df_grouped[f"Current_{dust_level}"] = self.df_grouped["Current_avg"] * (1 - d_I / 100)
            self.df_grouped[f"Power_{dust_level}"]   = self.df_grouped["Power_avg"]   * (1 - d_P / 100)

    def get_grouped_data(self):
        return self.df_grouped

#class ML:
    
class DataPlotter:

    def __init__(self, df, df_grouped, diff_val):
        self.df = df
        self.df_grouped = df_grouped
        self.diff_val = diff_val

    def plot(self):

        for dust_level, diff in self.diff_val.items():
            print(f"\nLevel of Dust {dust_level}:")
            print("Time | Voltage | Current | Power")
            for i, row in self.df_grouped.iterrows():
                if row['Time'].strftime('%H:%M') in self.df['Time'].dt.strftime('%H:%M').values:
                    print(f"{row['Time'].strftime('%H:%M')} | "
                          f"{row[f'Voltage_{dust_level}']:.2f} | "
                          f"{row[f'Current_{dust_level}']:.2f} | "
                          f"{row[f'Power_{dust_level}']  :.2f}")

        fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # สร้าง 3 แถว 1 คอลัมน์
        ax1, ax2, ax3 = axes  

        # Plot Voltage vs Time
        ax2.plot(self.df_grouped["Time_float"], self.df_grouped["Voltage_avg"], marker="o", linestyle="-", color="g", label="Average Voltage (V)")
        for dust_level, diff in self.diff_val.items():
            ax2.plot(self.df_grouped["Time_float"], self.df_grouped[f"Voltage_{dust_level}"], marker="o", linestyle=":", label=f"Voltage (Dust {dust_level})")
        ax2.set_xlabel("Time (Hour)")
        ax2.set_ylabel("Voltage (V)")
        ax2.set_title("Voltage vs Time")
        ax2.legend()
        ax2.grid()

        # Plot Current vs Time
        ax3.plot(self.df_grouped["Time_float"], self.df_grouped["Current_avg"], marker="o", linestyle="-", color="r", label="Average Current (A)")
        for dust_level, diff in self.diff_val.items():
            ax3.plot(self.df_grouped["Time_float"], self.df_grouped[f"Current_{dust_level}"], marker="o", linestyle=":", label=f"Current (Dust {dust_level})")
        ax3.set_xlabel("Time (Hour)")
        ax3.set_ylabel("Current (A)")
        ax3.set_title("Current vs Time")
        ax3.legend()
        ax3.grid()

         # Plot Power vs Time
        ax1.plot(self.df_grouped["Time_float"], self.df_grouped["Power_avg"], marker="o", linestyle="-", color="b", label="Average Power (W)")
        for dust_level, diff in self.diff_val.items():
            ax1.plot(self.df_grouped["Time_float"], self.df_grouped[f"Power_{dust_level}"], marker="o", linestyle=":", label=f"Power (Dust {dust_level})")
        ax1.set_xlabel("Time (Hour)")
        ax1.set_ylabel("Power (W)")
        ax1.set_title("Power vs Time")
        ax1.legend()
        ax1.grid()

        plt.tight_layout()
        plt.show()

class DataProcessor:
    
    def __init__(self, url_1, url_2):
        self.url_1 = url_1
        self.url_2 = url_2
        self.data_loader = DataLoader(self.url_2, self.url_1)
        self.data_analyzer = DataAnalyzer(self.data_loader.df, self.data_loader.diff_val)
        self.data_plotter = None

    def process(self):
        self.data_analyzer.calculate_average()
        self.data_analyzer.calculate_difference()  
        new_values = self.data_analyzer.get_grouped_data()
    #    print(new_values)
        self.data_plotter = DataPlotter(self.data_loader.df, new_values, self.data_loader.diff_val)
        self.data_plotter.plot()


# Sheet ID & Name
SPREADSHEET_ID = "11-f8Pye6G5vKX6xz3qr_EdeGjzZvyntvsBsvcBM8FX8"
SHEET_NAME_1 = "Sheet1" 
SHEET_NAME_2 = "Sheet2"

# Load data
url_1 = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME_1}"
url_2 = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME_2}"

# Object
data_processor = DataProcessor(url_1, url_2)
data_processor.process()
