import os
import sys
import json
import datetime
import torch  # 必须导入 torch
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify

# --- 1. 路径修正 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# --- 2. 导入模型并初始化 ---
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    
    # 初始化模型实例
    lan_engine = BuildTechGNN(input_dim=5)
    wan_engine = WAN_GNN(input_dim=18)
    
    # 设置为评估模式
    lan_engine.eval()
    wan_engine.eval()
    print("✅ [AI] LAN (5-dim) & WAN (18-dim) Engines Ready")
except Exception as e:
    print(f"❌ [AI] Model Load Error: {e}")
    lan_engine, wan_engine = None, None

app = Flask(__name__)

# ==========================================
#  3. LAN 处理器 (修正调用方式)
# ==========================================
def handle_lan_data(event):
    if event.data is None or lan_engine is None: return
    try:
        data = event.data
        sensors = data.get('sensors', {})
        weather = data.get('weather', {})
        
        # 准备 5 个输入特征: [mq135, mq2, mq7, temp, humidity]
        features = [
            float(sensors.get('mq135', {}).get('raw', 0)),
            float(sensors.get('mq2', {}).get('raw', 0)),
            float(sensors.get('mq7', {}).get('raw', 0)),
            float(weather.get('temp', 25)),
            float(weather.get('humidity', 50))
        ]
        
        # PyTorch 预测逻辑：转换为 Tensor -> 关闭梯度计算 -> 获取结果
        input_tensor = torch.FloatTensor([features])
        with torch.no_grad():
            prediction = lan_engine(input_tensor)
            lan_pred = prediction.item() # 获取单数值
        
        status = "Normal"
        if lan_pred > 15: status = "Warning"
        if lan_pred > 25: status = "Danger"

        firebase_db.reference("56214328/ai_analysis").update({
            "current_prediction": round(float(lan_pred), 4),
            "engine": "BuildTech-LAN-GNN-v2",
            "last_calc_time": datetime.datetime.utcnow().isoformat() + "Z",
            "status": status
        })
        print(f"🟢 [LAN AI] Result: {lan_pred:.2f}")
    except Exception as e:
        print(f"❌ [LAN AI] Inference Error: {e}")

# ==========================================
#  4. WAN 处理器 (修正调用方式)
# ==========================================
def handle_wan_data(event):
    if event.data is None or wan_engine is None: return
    try:
        readings = event.data.get('readings', {})
        # 你的 WAN 模型需要 18 个特征
        features = [float(val) for val in readings.values()]
        
        if len(features) == 18:
            input_tensor = torch.FloatTensor([features])
            with torch.no_grad():
                wan_pred = wan_engine(input_tensor).item()
            
            firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
                "territory_avg_prediction": round(float(wan_pred), 4),
                "last_calc_time": datetime.datetime.utcnow().isoformat() + "Z",
                "status": "Moderate" if wan_pred < 7 else "High Risk"
            })
            print(f"🔵 [WAN AI] Result: {wan_pred:.2f}")
    except Exception as e:
        print(f"❌ [WAN AI] Inference Error: {e}")

# ==========================================
#  5. 初始化与监听器
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
    
    firebase_db.reference("56214328/latest").listen(handle_lan_data)
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)
    print("📡 [System] Listeners Active")

@app.route('/')
def home():
    return jsonify({"status": "AI Engine Live", "pytorch_version": torch.__version__})

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
