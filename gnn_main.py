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

# --- 1. 路徑修正與模型導入 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    
    # 初始化模型
    lan_engine = BuildTechGNN()
    wan_engine = WAN_GNN()
    
    # 載入權重（如有保存的 .pth 文件，請取消下方註解）
    # lan_engine.load_state_dict(torch.load('models/lan_weight.pth'))
    # wan_engine.load_state_dict(torch.load('models/wan_weight.pth'))
    
    lan_engine.eval()
    wan_engine.eval()
    print("✅ [AI Engine] LAN 實時模型 & WAN 模擬模型加載成功")
except Exception as e:
    print(f"❌ [AI Engine] 模型初始化失敗: {e}")
    lan_engine, wan_engine = None, None

app = Flask(__name__)

# ==========================================
#  2. LAN 處理器：ESP32 實時數據分析
# ==========================================
def handle_lan_data(event):
    if event.data is None or lan_engine is None: return
    try:
        data = event.data
        sensors = data.get('sensors', {})
        weather = data.get('weather', {})
        
        # 特徵提取: [mq135, mq2, mq7, temp, humidity]
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
        
        # 狀態判定
        status = "Normal"
        if lan_pred > 15: status = "Warning"
        if lan_pred > 25: status = "Danger"

        # 回寫 LAN 分析結果
        firebase_db.reference("56214328/ai_analysis").update({
            "current_prediction": round(float(lan_pred), 4),
            "engine": "BuildTech-LAN-GNN",
            "last_calc_time": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            "status": status
        })
        print(f"🟢 [LAN] 實時 AQI 預測: {lan_pred:.2f} ({status})")
    except Exception as e:
        print(f"❌ [LAN] 處理錯誤: {e}")

# ==========================================
#  3. WAN 處理器：18區數據處理 + 6小時模擬
# ==========================================
def handle_wan_data(event):
    if event.data is None or wan_engine is None: return
    try:
        readings = event.data.get('readings', {})
        # 按順序提取 18 區數值
        features = [float(val) for val in readings.values()]
        
        if len(features) == 18:
            # --- A. 實時推理 (Real-time) ---
            input_tensor = torch.FloatTensor([features])
            with torch.no_grad():
                current_wan_pred = wan_engine(input_tensor).item()
            
            # --- B. 模擬 6 小時後數據 (Simulation) ---
            # 1. 模擬 18 區每個站點獨立的趨勢變化 (波動幅度 ±10%)
            sim_readings = {}
            stations = list(readings.keys())
            for station in stations:
                original_val = float(readings[station])
                # 模擬環境變化因子
                drift = random.uniform(0.9, 1.1) 
                sim_readings[station] = round(original_val * drift, 1)
            
            # 2. 將模擬出的 18 區數據再次輸入 GNN 模型
            sim_features = [float(v) for v in sim_readings.values()]
            sim_tensor = torch.FloatTensor([sim_features])
            with torch.no_grad():
                future_wan_pred = wan_engine(sim_tensor).item()
            
            # 3. 設定目標時間 (現在 + 6小時)
            now = datetime.datetime.now(datetime.UTC)
            target_time = (now + timedelta(hours=6)).isoformat() + "Z"

            # --- C. 回寫 WAN 分析結果 ---
            firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
                "territory_avg_prediction": round(float(current_wan_pred), 4),
                "simulation_6h_later": round(float(future_wan_pred), 4),
                "simulated_18_districts": sim_readings, # 存儲 18 區模擬值供前端繪圖
                "simulation_target_time": target_time,
                "engine": "BuildTech-WAN-GNN-v2-Sim",
                "last_calc_time": now.isoformat() + "Z",
                "status": "Moderate" if current_wan_pred < 7 else "High Risk"
            })
            print(f"🔵 [WAN] 當前全港平均: {current_wan_pred:.2f} | 6h 預測模擬: {future_wan_pred:.2f}")

    except Exception as e:
        print(f"❌ [WAN] 處理錯誤: {e}")

# ==========================================
#  4. 服務初始化與啟動
# ==========================================
def start_services():
    # Firebase 認證處理
    fb_config_str = os.environ.get("FIREBASE_CONFIG")
    if fb_config_str:
        cred = credentials.Certificate(json.loads(fb_config_str, strict=False))
    else:
        # 本地開發路徑
        cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        cred = credentials.Certificate(cred_path)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
    
    # 註冊 Firebase 監聽器
    firebase_db.reference("56214328/latest").listen(handle_lan_data)
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)
    print("📡 [System] 雙模監聽器已激活：LAN (實時) / WAN (6h 預測)")

@app.route('/')
def home():
    return jsonify({
        "status": "Running",
        "features": ["LAN Real-time Analysis", "WAN 6H Simulation Forecasting"],
        "timestamp": datetime.datetime.now().isoformat()
    })

if __name__ == "__main__":
    start_services()
    # 支援 Render/Heroku 等雲平台的端口綁定
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
