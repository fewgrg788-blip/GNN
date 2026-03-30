import os
import sys
import json
import datetime
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

# --- 导入双模型 ---
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    lan_engine = BuildTechGNN() # 假设这是你的 LAN 模型初始化
    wan_engine = WAN_GNN()      # 假设这是你的 WAN 模型初始化
    print("✅ [AI] Both LAN & WAN Engines Loaded")
except Exception as e:
    print(f"❌ [AI] Engine Load Error: {e}")
    lan_engine, wan_engine = None, None

app = Flask(__name__)

# ==========================================
#  1. LAN 数据处理器 (微观环境 - ESP32)
# ==========================================
def handle_lan_data(event):
    if event.data is None or lan_engine is None: return
    try:
        data = event.data
        # 解析你在 JSON 中定义的结构
        mq135 = data.get('sensors', {}).get('mq135', {}).get('raw', 0)
        mq2 = data.get('sensors', {}).get('mq2', {}).get('raw', 0)
        mq7 = data.get('sensors', {}).get('mq7', {}).get('raw', 0)
        temp = data.get('weather', {}).get('temp', 25)
        hum = data.get('weather', {}).get('humidity', 50)
        
        # 运行 LAN 预测
        lan_pred = lan_engine.predict([mq135, mq2, mq7, temp, hum])
        
        status = "Normal"
        if lan_pred > 15: status = "Warning"
        if lan_pred > 25: status = "Danger"

        # 按照格式写回 Firebase
        firebase_db.reference("56214328/ai_analysis").update({
            "current_prediction": round(float(lan_pred), 4),
            "engine": "BuildTech-LAN-GNN-v2",
            "last_calc_time": datetime.datetime.utcnow().isoformat() + "Z",
            "status": status,
            "trigger_source": "MQ_Sensors"
        })
        print(f"🟢 [LAN AI] Updated: {lan_pred:.2f} ({status})")
    except Exception as e:
        print(f"❌ [LAN AI] Error: {e}")

# ==========================================
#  2. WAN 数据处理器 (宏观环境 - 18区数据)
# ==========================================
def handle_wan_data(event):
    if event.data is None or wan_engine is None: return
    try:
        readings = event.data.get('readings', {})
        
        # 提取 18 区的值转换为列表 (Causeway Bay, Central, etc.)
        # 你的 JSON 中值是字符串 "4", "2", 需要转为整数或浮点数
        features = [float(val) for key, val in readings.items()]
        
        # 运行 WAN 预测 (例如预测未来一小时的全港平均 AQHI)
        wan_pred = wan_engine.predict(features)
        
        status = "Good"
        if wan_pred >= 4: status = "Moderate"
        if wan_pred >= 7: status = "High Risk"

        # 写回 GAGNN_24hours 节点
        firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
            "territory_avg_prediction": round(float(wan_pred), 4),
            "engine": "BuildTech-WAN-GNN-v1",
            "last_calc_time": datetime.datetime.utcnow().isoformat() + "Z",
            "status": status,
            "trigger_source": "HK_Gov_API"
        })
        print(f"🔵 [WAN AI] Updated: {wan_pred:.2f} ({status})")
    except Exception as e:
        print(f"❌ [WAN AI] Error: {e}")


# ==========================================
#  启动与路由
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
    
    # 建立双通道监听 (Dual-Channel Listeners)
    firebase_db.reference("56214328/latest").listen(handle_lan_data)
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)
    print("📡 [System] Dual-Channel Listeners (LAN & WAN) Started")

@app.route('/')
def home():
    return jsonify({
        "status": "AI Engines Online", 
        "modules": ["LAN_GNN", "WAN_GNN"],
        "listeners": "Active"
    })

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
