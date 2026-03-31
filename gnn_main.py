import os
import sys
import json
import random
import datetime
from datetime import timedelta
import torch
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify

# --- 1. 路徑與環境設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# --- 2. 加載 LAN & WAN 模型 ---
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    
    lan_engine = BuildTechGNN()
    wan_engine = WAN_GNN()
    
    lan_engine.eval()
    wan_engine.eval()
    print("✅ [AI] 模型加載完成。WAN 系統已配置 6 小時模擬預測。")
except Exception as e:
    print(f"❌ [AI] 模型加載失敗: {e}")
    lan_engine, wan_engine = None, None

app = Flask(__name__)

# ==========================================
#  3. LAN 處理器 (保持原樣：僅處理實時數據)
# ==========================================
def handle_lan_data(event):
    if event.data is None or lan_engine is None: return
    try:
        data = event.data
        sensors = data.get('sensors', {})
        weather = data.get('weather', {})
        
        features = [
            float(sensors.get('mq135', {}).get('raw', 0)),
            float(sensors.get('mq2', {}).get('raw', 0)),
            float(sensors.get('mq7', {}).get('raw', 0)),
            float(weather.get('temp', 25)),
            float(weather.get('humidity', 50))
        ]
        
        input_tensor = torch.FloatTensor([features])
        with torch.no_grad():
            lan_pred = lan_engine(input_tensor).item() 
        
        status = "Normal"
        if lan_pred > 15: status = "Warning"
        if lan_pred > 25: status = "Danger"

        firebase_db.reference("56214328/ai_analysis").update({
            "current_prediction": round(float(lan_pred), 4),
            "engine": "BuildTech-LAN-GNN",
            "last_calc_time": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            "status": status
        })
        print(f"🟢 [LAN] 實時分析完成: {lan_pred:.2f}")
    except Exception as e:
        print(f"❌ [LAN] 錯誤: {e}")

# ==========================================
#  4. WAN 處理器 (加入 6 小時模擬預測)
# ==========================================
def handle_wan_data(event):
    if event.data is None or wan_engine is None: return
    try:
        # 獲取 18 區原始數據
        readings = event.data.get('readings', {})
        features = [float(val) for val in readings.values()]
        
        if len(features) == 18:
            # A. 實時預測
            input_tensor = torch.FloatTensor([features])
            with torch.no_grad():
                current_wan_pred = wan_engine(input_tensor).item()
            
            # B. 模擬 6 小時後的趨勢
            # 模擬 18 區數據在未來 6 小時的輕微變動 (±5%)
            future_features = [f * random.uniform(0.95, 1.05) for f in features]
            future_tensor = torch.FloatTensor([future_features])
            with torch.no_grad():
                future_wan_pred = wan_engine(future_tensor).item()
            
            # C. 準備時間軸數據
            now = datetime.datetime.now(datetime.UTC)
            forecast_time = (now + timedelta(hours=6)).isoformat() + "Z"
            
            status = "Moderate" if current_wan_pred < 7 else "High Risk"

            # 寫入 Firebase
            firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
                "territory_avg_prediction": round(float(current_wan_pred), 4),
                "simulation_6h_later": round(float(future_wan_pred), 4),
                "simulation_target_time": forecast_time,
                "engine": "BuildTech-WAN-GNN-SimMode",
                "last_calc_time": now.isoformat() + "Z",
                "status": status
            })
            print(f"🔵 [WAN] 當前平均: {current_wan_pred:.2f} | 6小時模擬結果: {future_wan_pred:.2f}")
            
    except Exception as e:
        print(f"❌ [WAN] 運行時錯誤: {e}")

# ==========================================
#  5. 服務啟動
# ==========================================
def start_services():
    # 這裡會自動讀取你設定的環境變量或本地 Key 文件
    fb_config_str = os.environ.get("FIREBASE_CONFIG")
    if fb_config_str:
        cred = credentials.Certificate(json.loads(fb_config_str, strict=False))
    else:
        cred = credentials.Certificate(os.path.join(BASE_DIR, "serviceAccountKey.json"))

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
    
    # 分別監聽 LAN 和 WAN 的路徑
    firebase_db.reference("56214328/latest").listen(handle_lan_data)
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)
    print("📡 [System] 雙系統監聽已啟動 (WAN 預測模式開啟)")

@app.route('/')
def home():
    return jsonify({
        "status": "Online", 
        "wan_mode": "Simulating 6 hours later",
        "lan_mode": "Real-time"
    })

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
