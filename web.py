from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# --- เพิ่มไลบรารีสำหรับดึงข้อมูลจากเว็บ ---
import requests
from bs4 import BeautifulSoup

# ... (โค้ดเดิมทั้งหมด) ...

class DataProcessor:
      #...
      def process(self):
          # ... (โค้ดเดิม)
          print("\nData from the database:")
          for row in self.db.fetch_all_data():
              print(row)
          # เพิ่มฟังก์ชั้น fetch_from_database
          self.fetch_data_from_database_by_time()

      def fetch_data_from_database_by_time(self):
          while True:
              search_time = input("Enter Time to search in database (HH:MM) or 'e' to exit: ")
              if search_time.lower() == 'e':
                  break
              
              self.db.cursor.execute("SELECT * FROM energy_data WHERE time=?", (search_time,))
              result = self.db.cursor.fetchone()
              if result:
                  print("Found data:")
                  print(result)
              else:
                  print(f"No data found for time: {search_time}")
          
# --- เพิ่มฟังก์ชัน fetch_data_from_web ---
def fetch_data_from_web(url):
    """
    ดึงข้อมูลเวลาและระดับฝุ่นจากเว็บที่กำหนด

    Args:
        url (str): URL ของเว็บที่ต้องการดึงข้อมูล

    Returns:
        tuple: (time_input, dust_level_input) หากดึงข้อมูลสำเร็จ, (None, None) หากเกิดข้อผิดพลาด
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # ตรวจสอบว่าการร้องขอสำเร็จหรือไม่

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- ปรับแต่งส่วนนี้ตามโครงสร้าง HTML ของเว็บที่คุณต้องการดึงข้อมูล ---
        # ตัวอย่าง: ดึงข้อมูลจาก <div id="time">10:00</div> และ <div id="dust">2.5</div>
        time_element = soup.find('div', id='time')
        dust_element = soup.find('div', id='dust')

        if time_element and dust_element:
            time_input = time_element.text.strip()
            dust_level_input = float(dust_element.text.strip())
            return time_input, dust_level_input
        else:
            print("ไม่พบข้อมูลเวลาหรือระดับฝุ่นในเว็บ")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"เกิดข้อผิดพลาดในการดึงข้อมูลจากเว็บ: {e}")
        return None, None
    except ValueError:
        print("ข้อมูลระดับฝุ่นที่ดึงมาไม่ถูกต้อง ควรเป็นตัวเลข")
        return None, None

# URL
SPREADSHEET_ID = "11-f8Pye6G5vKX6xz3qr_EdeGjzZvyntvsBsvcBM8FX8"
SHEET_NAME_1 = "Sheet1" 
SHEET_NAME_2 = "Sheet2"

url_1 = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME_1}"
url_2 = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME_2}"

# Object
data_processor = DataProcessor(url_1, url_2)
data_processor.process()

# --- ปรับเปลี่ยน Loop 13 ---
# --- กำหนด URL ของเว็บที่ต้องการดึงข้อมูล ---
WEB_DATA_URL = "http://yourwebsite.com/data"  # <--- เปลี่ยนตรงนี้เป็น URL จริงของคุณ

while True:
    # --- ดึงข้อมูลจากเว็บ ---
    time_input, dust_level_input = fetch_data_from_web(WEB_DATA_URL)

    if time_input is None or dust_level_input is None:
        print("ไม่สามารถดึงข้อมูลจากเว็บได้ หรือข้อมูลไม่ถูกต้อง")
        user_choice = input("คุณต้องการลองใหม่ (r) หรือออกจากโปรแกรม (e)? : ")
        if user_choice.lower() == 'e':
             break
        else:
             continue
    else:
        print(f"ดึงข้อมูลจากเว็บสำเร็จ: เวลา {time_input}, ระดับฝุ่น {dust_level_input}")
        
        new_data = NewData(time_input, dust_level_input)
        new_data.predictNew(data_processor)