import os
import torch
import datetime
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask

# 從自定義模組導入模型類 [cite: 1]
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    print("✅ [System] Models modules imported successfully.")
except ImportError as e:
    print(f"❌ [System] Import Error: {e}")

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 系統狀態追蹤
STATUS = {
    "firebase": "Initializing...",
    "lan_model": "Initial",
    "wan_model": "Initial",
    "last_run": "None"
}

# --- 1. 初始化模型實例 ---
# 確保維度與你訓練 .pth 時一致 [cite: 1]
lan_engine = BuildTechGNN(input_dim=5) 
wan_engine = WAN_GNN(input_dim=18)

# --- 2. 核心預測邏輯 ---
def handle_prediction(event):
    path = event.path
    data = event.data
    if not data:
        return

    now_str = datetime.datetime.now().isoformat()
    print(f"\n--- [Event] Data change detected at {path} ---")

    try:
        # 🟢 情況 A: 處理 LAN 數據 (來自 ESP32 傳感器)
        if "sensors" in path or ("latest" in path and data.get("sensors")):
            print("🔍 [LAN] Extracting sensor features...")
            sensors = data.get('sensors', data)
            weather = data.get('weather', {})
            
            # 構建特徵向量 [mq135, mq2, mq7, temp, hum]
            features = [
                sensors.get('mq135', {}).get('norm', 0),
                sensors.get('mq2', {}).get('norm', 0),
                sensors.get('mq7', {}).get('norm', 0),
                weather.get('temp', 25),
                weather.get('humidity', 50)
            ]
            
            x = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                score = lan_engine(x).item()
            
            print(f"🎯 [LAN] Prediction Score: {score:.4f}")
            
            # 寫回 Firebase
            db.reference('56214328/ai_analysis').update({
                "current_prediction": round(score, 4),
                "status": "Warning" if score > 0.5 else "Safe",
                "last_calc_time": now_str,
                "engine": "BuildTech-LAN-GNN"
            })

        # 🔵 情況 B: 處理 WAN 數據 (來自 GitHub Action 的全港 API)
        elif "latest_api" in path:
            print("🔍 [WAN] Extracting Hong Kong station features...")
            # 假設數據是 18 個站點的數值字典
            features = [float(v) for v in data.values()]
            
            if len(features) == 18:
                x = torch.tensor([features], dtype=torch.float32)
                with torch.no_grad():
                    score = wan_engine(x).item()
                
                print(f"🎯 [WAN] Regional Score: {score:.4f}")
                db.reference('56214328/regional_analysis').set({
                    "score": round(score, 4),
                    "timestamp": now_str,
                    "engine": "BuildTech-WAN-GNN"
                })
            else:
                print(f"⚠️ [WAN] Data dimension mismatch. Expected 18, got {len(features)}")

        STATUS["last_run"] = now_str

    except Exception as e:
        print(f"❌ [Error] Prediction Loop Failed: {e}")

# --- 3. 系統啟動函數 ---
def start_system():
    print("🚀 [System] Starting BuildTech GNN Engine...")

    # A. 載入權重
    paths = {
        "lan": os.path.join(BASE_DIR, "weights", "model_lan.pth"),
        "wan": os.path.join(BASE_DIR, "weights", "hk_pro_model_final.pth")
    }

    for key, p in paths.items():
        if os.path.exists(p):
            try:
                model = lan_engine if key == "lan" else wan_engine
                model.load_state_dict(torch.load(p, map_location='cpu'))
                model.eval()
                STATUS[f"{key}_model"] = "Loaded Successfully"
                print(f"📦 [System] {key.upper()} weights loaded from {p}")
            except Exception as e:
                STATUS[f"{key}_model"] = f"Load Error: {e}"
                print(f"❌ [System] Failed to load {key} weights: {e}")
        else:
            STATUS[f"{key}_model"] = "File not found"
            print(f"⚠️ [System] Weight file missing: {p}")

    # B. 連接 Firebase [cite: 1]
    try:
        cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        STATUS["firebase"] = "Connected"
        print("🔥 [System] Firebase real-time listener active.")
        
        # 監聽整個項目根目錄 [cite: 1]
        db.reference('56214328').listen(handle_prediction)
    except Exception as e:
        STATUS["firebase"] = f"Connection Failed: {e}"
        print(f"❌ [System] Firebase Error: {e}")

@app.route('/')
def health_check():
    return {
        "engine_status": STATUS,
        "server_time": datetime.datetime.now().isoformat()
    }, 200

if __name__ == '__main__':
    start_system()
    # Render 環境變量端口 [cite: 1]
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
