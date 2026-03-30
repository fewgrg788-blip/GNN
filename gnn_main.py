import os
import sys
import json
import datetime
import torch  # 必须确保导入了 torch
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify

# --- 1. 路径修正 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# --- 2. 加载模型 ---
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    
    # 初始化模型实例 (根据你的源代码，lan 默认 input_dim=5, wan 默认 18)
    lan_engine = BuildTechGNN()
    wan_engine = WAN_GNN()
    
    # 切换到评估模式
    lan_engine.eval()
    wan_engine.eval()
    print("✅ [AI] LAN & WAN 模型已加载并进入 Eval 模式")
except Exception as e:
    print(f"❌ [AI] 加载模型失败: {e}")
    lan_engine, wan_engine = None, None

app = Flask(__name__)

# ==========================================
#  3. LAN 处理器 (修复 predict 报错)
# ==========================================
def handle_lan_data(event):
    if event.data is None or lan_engine is None: return
    try:
        data = event.data
        sensors = data.get('sensors', {})
        weather = data.get('weather', {})
        
        # 准备 5 个输入特征：[mq135, mq2, mq7, temp, humidity]
        features = [
            float(sensors.get('mq135', {}).get('raw', 0)),
            float(sensors.get('mq2', {}).get('raw', 0)),
            float(sensors.get('mq7', {}).get('raw', 0)),
            float(weather.get('temp', 25)),
            float(weather.get('humidity', 50))
        ]
        
        # --- 核心修复：使用 PyTorch 的 __call__ 而不是 .predict() ---
        input_tensor = torch.FloatTensor([features])
        with torch.no_grad():
            output = lan_engine(input_tensor) # 直接调用模型对象
            lan_pred = output.item() 
        
        # 判断状态
        status = "Normal"
        if lan_pred > 15: status = "Warning"
        if lan_pred > 25: status = "Danger"

        # 写回 Firebase
        firebase_db.reference("56214328/ai_analysis").update({
            "current_prediction": round(float(lan_pred), 4),
            "engine": "BuildTech-LAN-GNN-v2-Fixed",
            "last_calc_time": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            "status": status
        })
        print(f"🟢 [LAN AI] 预测成功: {lan_pred:.4f}")

    except Exception as e:
        print(f"❌ [LAN AI] 运行时错误: {e}")

# ==========================================
#  4. WAN 处理器
# ==========================================
def handle_wan_data(event):
    if event.data is None or wan_engine is None: return
    try:
        readings = event.data.get('readings', {})
        features = [float(val) for val in readings.values()]
        
        if len(features) == 18:
            input_tensor = torch.FloatTensor([features])
            with torch.no_grad():
                wan_pred = wan_engine(input_tensor).item()
            
            firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
                "territory_avg_prediction": round(float(wan_pred), 4),
                "last_calc_time": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                "status": "Moderate" if wan_pred < 7 else "High Risk"
            })
            print(f"🔵 [WAN AI] 预测成功: {wan_pred:.4f}")
    except Exception as e:
        print(f"❌ [WAN AI] 运行时错误: {e}")

# ==========================================
#  5. 服务启动
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
    
    # 注册监听器
    firebase_db.reference("56214328/latest").listen(handle_lan_data)
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)
    print("📡 [System] 实时监听通道已激活")

@app.route('/')
def home():
    return jsonify({"status": "Online", "engines": "LAN & WAN Active"})

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
