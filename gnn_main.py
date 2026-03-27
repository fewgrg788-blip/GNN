import os
import torch
import datetime
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask

# 从 models 文件夹导入你的模型类
from models.lan_gnn import BuildTechGNN
from models.wan_gnn import WAN_GNN

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS = {"firebase": "Initializing...", "lan_model": "Initial", "wan_model": "Initial"}

# --- 1. 初始化模型实例 ---
# 假设 LAN 输入 5 维，WAN 输入 18 维（请根据你的训练参数调整）
lan_engine = BuildTechGNN(input_dim=5)
wan_engine = WAN_GNN(input_dim=18)

# --- 2. 核心计算分发器 ---
def handle_prediction(event):
    path = event.path
    data = event.data
    if not data: return

    try:
        # 情况 A: 处理来自硬件 (ESP32) 的 LAN 数据
        if "latest_hardware" in path or "sensors" in str(path):
            print("🔌 Processing LAN Data (MQ135/Sensors)...")
            # 提取数据 (参考你之前的逻辑)
            sensors = data.get('sensors', data) # 兼容不同路径结构
            features = [
                sensors.get('mq135', {}).get('norm', 0),
                sensors.get('mq2', {}).get('norm', 0),
                sensors.get('mq7', {}).get('norm', 0),
                sensors.get('temp', 25) / 50.0,
                sensors.get('hum', 50) / 100.0
            ]
            input_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                score = lan_engine(input_tensor).item()
            
            db.reference('56214328/results/lan_score').set({
                "score": round(score, 4),
                "time": datetime.datetime.now().isoformat()
            })

        # 情况 B: 处理来自 API 的 WAN 数据 (全港)
        elif "latest_api" in path:
            print("🛰️ Processing WAN Data (Hong Kong AQHI)...")
            # 假设数据是 18 个站点的数值列表
            features = list(data.values()) 
            input_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                score = wan_engine(input_tensor).item()
            
            db.reference('56214328/results/wan_score').set({
                "score": round(score, 4),
                "time": datetime.datetime.now().isoformat()
            })

    except Exception as e:
        print(f"❌ Prediction Error: {e}")

# --- 3. 系统启动与权重加载 ---
def start_system():
    # 載入 LAN 權重
    lan_weight = os.path.join(BASE_DIR, "weights", "model_lan.pth")
    if os.path.exists(lan_weight):
        try:
            lan_engine.load_state_dict(torch.load(lan_weight, map_location='cpu'))
            lan_engine.eval()
            STATUS["lan_model"] = "Loaded"
        except Exception as e: STATUS["lan_model"] = f"Error: {e}"

    # 載入 WAN 權重
    wan_weight = os.path.join(BASE_DIR, "weights", "hk_pro_model_final.pth")
    if os.path.exists(wan_weight):
        try:
            wan_engine.load_state_dict(torch.load(wan_weight, map_location='cpu'))
            wan_engine.eval()
            STATUS["wan_model"] = "Loaded"
        except Exception as e: STATUS["wan_model"] = f"Error: {e}"

    # Firebase 連接
    try:
        cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        STATUS["firebase"] = "Connected"
        # 監聽整個項目路徑以捕捉不同類型的數據更新
        db.reference('56214328').listen(handle_prediction)
    except Exception as e:
        STATUS["firebase"] = f"Failed: {e}"

@app.route('/')
def health_check():
    return {
        "status": "BuildTech GNN Engine Running",
        "models": { "LAN": STATUS["lan_model"], "WAN": STATUS["wan_model"] },
        "firebase": STATUS["firebase"]
    }

if __name__ == '__main__':
    start_system()
    app.run(host='0.0.0.0', port=10000)
