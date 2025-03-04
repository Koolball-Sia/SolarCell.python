from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
          df = pd.read_csv(self.url_2, encoding="utf-8")
          df.columns = df.columns.str.strip()  # Clean column names
          df["Time"] = pd.to_datetime(df["Time"], format="%H:%M")  # Convert time to datetime
          df["Time_float"] = df["Time"].dt.hour + df["Time"].dt.minute / 60  # Convert time to float for plotting
          return df

      def load_diff_val(self):
          df_1 = pd.read_csv(self.url_1, encoding="utf-8")
          dust_diff_map = {}

          for _, row in df_1.iterrows():
              dust_level = row["level of dust"]
              dust_diff_map[dust_level] = { 'V': row["%Diff_V"],
                                            'I': row["%Diff_I"],
                                            'P': row["%Diff_P"] }
          return dust_diff_map

class DataAnalyzer:

      def __init__(self, df, diff_val):
          self.df = df
          self.df_grouped = None
          self.diff_val = diff_val

      def calculate_average(self):
          self.df_grouped = self.df.groupby('Time').agg(
              Power_avg   = ('Power(W)', 'mean'),
              Voltage_avg = ('Voltage(V)', 'mean'),
              Current_avg = ('Current(I)', 'mean')  ).reset_index()
        
          self.df_grouped["Time_float"] = self.df_grouped["Time"].dt.hour + self.df_grouped["Time"].dt.minute / 60

      def calculate_difference(self):
          for dust_level, diff in self.diff_val.items():
              d_V, d_I, d_P = diff["V"], diff["I"], diff["P"]
            
              self.df_grouped[f"Voltage_{dust_level}"] = self.df_grouped["Voltage_avg"] * (1 - d_V / 100)
              self.df_grouped[f"Current_{dust_level}"] = self.df_grouped["Current_avg"] * (1 - d_I / 100)
              self.df_grouped[f"Power_{dust_level}"]   = self.df_grouped["Power_avg"]   * (1 - d_P / 100)

      def get_grouped_data(self):
          return self.df_grouped

class DataList:
      
      def __init__(self, df_grouped, diff_val):
          self.df_grouped = df_grouped
          self.diff_val = diff_val

      def list_data(self):
          Time = []
          Dust = []
          V = []
          I = []
          P = []
        
          for dust_level in self.diff_val.keys():
              for _, row in self.df_grouped.iterrows():
                  Time.append(row['Time'].strftime('%H:%M'))  
                  Dust.append(dust_level)                    
                  V.append(f"{row[f'Voltage_{dust_level}']:.2f}")  
                  I.append(f"{row[f'Current_{dust_level}']:.2f}") 
                  P.append(f"{row[f'Power_{dust_level}']:.2f}")    
                
          return Time, Dust, V, I, P

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
 
          fig, axes = plt.subplots(3, 1, figsize=(10, 15))  
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

class ML_LR:
    
      def __init__(self, X, y):
          self.X = X
          self.y = y
          self.model = LinearRegression()
    
      def train_model(self):
          self.model.fit(self.X, self.y)
    
      def predict(self, X_test):
          return self.model.predict(X_test)
    
      def evaluate(self, X_test, y_test):
          predictions = self.predict(X_test)
          mse = mean_squared_error(y_test, predictions)
          print(f'Mean Squared Error: {mse}')
          return mse
    
class DataProcessor:
    
      def __init__(self, url_1, url_2):
          self.url_1 = url_1
          self.url_2 = url_2
          self.data_loader = DataLoader(self.url_2, self.url_1)
          self.data_analyzer = DataAnalyzer(self.data_loader.df, self.data_loader.diff_val)
          self.list_data = None
          self.model = None

      def process(self):
          self.data_analyzer.calculate_average()
          self.data_analyzer.calculate_difference()
          self.list_data = DataList(self.data_analyzer.df_grouped, self.data_loader.diff_val)
        
          Time, Dust, V, I, P = self.list_data.list_data()
        
          X = np.array([[float(time.split(":")[0]) + float(time.split(":")[1])/60, dust] for time, dust in zip(Time, Dust)])
          y_V = np.array([float(voltage) for voltage in V])
          y_I = np.array([float(current) for current in I])
          y_P = np.array([float(power) for power in P])

          X_train, X_test, y_train_V, y_test_V = train_test_split(X, y_V, test_size=0.2, random_state=42)
          X_train, X_test, y_train_I, y_test_I = train_test_split(X, y_I, test_size=0.2, random_state=42)
          X_train, X_test, y_train_P, y_test_P = train_test_split(X, y_P, test_size=0.2, random_state=42)

          self.model_V = ML_LR(X_train, y_train_V)
          self.model_V.train_model()
          self.model_I = ML_LR(X_train, y_train_I)
          self.model_I.train_model()
          self.model_P = ML_LR(X_train, y_train_P)
          self.model_P.train_model()

          print("Voltage Model Evaluation:")
          self.model_V.evaluate(X_test, y_test_V)
          print("Current Model Evaluation:")
          self.model_I.evaluate(X_test, y_test_I)
          print("Power Model Evaluation:")
          self.model_P.evaluate(X_test, y_test_P)

class NewData:

      def __init__(self, time, dust_level):
          self.time = time  
          self.dust_level = dust_level 
          self.time_float = self.convert_time_to_float()

      def convert_time_to_float(self):
          hours, minutes = map(int, self.time.split(":"))
          return hours + minutes / 60

      def to_array(self):
          return np.array([[self.time_float, self.dust_level]])

      def predictNew(self, data_processor):
          X_new = self.to_array()  

          predicted_V = data_processor.model_V.predict(X_new)[0]
          predicted_I = data_processor.model_I.predict(X_new)[0]
          predicted_P = data_processor.model_P.predict(X_new)[0]

          print(f"Predicted Voltage: {predicted_V:.2f} V")
          print(f"Predicted Current: {predicted_I:.2f} A")
          print(f"Predicted Power: {predicted_P:.2f} W")

# URL
SPREADSHEET_ID = "11-f8Pye6G5vKX6xz3qr_EdeGjzZvyntvsBsvcBM8FX8"
SHEET_NAME_1 = "Sheet1" 
SHEET_NAME_2 = "Sheet2"

url_1 = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME_1}"
url_2 = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME_2}"

# Object
data_processor = DataProcessor(url_1, url_2)
data_processor.process()

new1 = NewData("14:15", 1.6)
new2 = NewData("13:08", 3.4)
new1.predictNew(data_processor)  
new2.predictNew(data_processor)  