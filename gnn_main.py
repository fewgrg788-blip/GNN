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

# --- 1. 路徑與模型加載 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    
    lan_engine = BuildTechGNN()
    wan_engine = WAN_GNN()
    
    lan_engine.eval()
    wan_engine.eval()
    print("✅ [AI] LAN (Real-time) & WAN (Simulation) 模型已就緒")
except Exception as e:
    print(f"❌ [AI] 加載模型失敗: {e}")
    lan_engine, wan_engine = None, None

app = Flask(__name__)

# ==========================================
#  3. LAN 處理器 (Real-time Only)
# ==========================================
def handle_lan_data(event):
    if event.data is None or lan_engine is None: return
    try:
        data = event.data
        sensors = data.get('sensors', {})
        weather = data.get('weather', {})
        
        # 特徵值: [mq135, mq2, mq7, temp, humidity]
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
        print(f"🟢 [LAN] 實時分析: {lan_pred:.2f}")

    except Exception as e:
        print(f"❌ [LAN] 運行錯誤: {e}")

# ==========================================
#  4. WAN 處理器 (Real-time + 6h Simulation)
# ==========================================
def handle_wan_data(event):
    if event.data is None or wan_engine is None: return
    try:
        readings = event.data.get('readings', {})
        features = [float(val) for val in readings.values()]
        
        if len(features) == 18:
            # A. 實時推理
            input_tensor = torch.FloatTensor([features])
            with torch.no_grad():
                current_wan_pred = wan_engine(input_tensor).item()
            
            # B. 模擬 6 小時後數據 (假設數據受環境影響產生 ±8% 的波動演變)
            sim_features = [f * random.uniform(0.92, 1.08) for f in features]
            sim_tensor = torch.FloatTensor([sim_features])
            with torch.no_grad():
                future_wan_pred = wan_engine(sim_tensor).item()
            
            now = datetime.datetime.now(datetime.UTC)
            target_time = (now + timedelta(hours=6)).isoformat() + "Z"

            # 寫入 Firebase
            firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
                "territory_avg_prediction": round(float(current_wan_pred), 4),
                "simulation_6h_later": round(float(future_wan_pred), 4),
                "simulation_target_time": target_time,
                "engine": "BuildTech-WAN-GNN-v2-Sim",
                "last_calc_time": now.isoformat() + "Z",
                "status": "Moderate" if current_wan_pred < 7 else "High Risk"
            })
            print(f"🔵 [WAN] 當前: {current_wan_pred:.2f} | 6小時後模擬: {future_wan_pred:.2f}")

    except Exception as e:
        print(f"❌ [WAN] 運行錯誤: {e}")

# ==========================================
#  5. 服務初始化
# ==========================================
def start_services():
    fb_config_str = os.environ.get("FIREBASE_CONFIG")
    if fb_config_str:
        cred = credentials.Certificate(json.loads(fb_config_str, strict=False))
    else:
        cred = credentials.Certificate(os.path.join(BASE_DIR, "serviceAccountKey.json"))

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
    
    # 同時監聽微觀 (LAN) 與 宏觀 (WAN)
    firebase_db.reference("56214328/latest").listen(handle_lan_data)
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)
    print("📡 [System] 雙模監聽中：LAN (實時) / WAN (預測模擬)")

@app.route('/')
def home():
    return jsonify({"status": "Online", "wan_simulation": "Active (6h)"})

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
