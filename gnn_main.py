import os
import sys
import json
import datetime
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify

# --- 1. 路径修正 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

print(f"🔍 [Debug] 当前工作目录: {BASE_DIR}")
print(f"🔍 [Debug] Python 路径: {sys.path[:3]}")

# --- 2. 导入模型 ---
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    lan_engine = BuildTechGNN()
    wan_engine = WAN_GNN()
    print("✅ [Debug] LAN & WAN 模型加载成功")
except Exception as e:
    print(f"❌ [Debug] 模型加载失败: {e}")
    lan_engine, wan_engine = None, None

app = Flask(__name__)

# ==========================================
#  3. LAN 逻辑 + 详细日志
# ==========================================
def handle_lan_data(event):
    print(f"🔔 [Event] LAN 节点更新! 路径: {event.path}")
    if event.data is None:
        print("⚠️ [Debug] LAN 收到空数据，跳过")
        return
    
    try:
        data = event.data
        print(f"📥 [Debug] LAN 原始数据: {json.dumps(data)[:100]}...") # 只打印前100字

        # 解析数据
        sensors = data.get('sensors', {})
        mq135 = sensors.get('mq135', {}).get('raw', 0)
        print(f"📊 [Debug] 解析 MQ135: {mq135}")

        # 模拟 AI 计算 (如果引擎没加载则用倍率模拟，确保能写回数据)
        if lan_engine:
            lan_pred = lan_engine.predict([mq135]) 
        else:
            print("⚠️ [Debug] LAN 引擎未就绪，使用模拟计算")
            lan_pred = mq135 * 0.05 

        # 写回 Firebase
        update_data = {
            "current_prediction": round(float(lan_pred), 4),
            "engine": "BuildTech-LAN-GNN-v2-Debug",
            "last_calc_time": datetime.datetime.utcnow().isoformat() + "Z",
            "status": "Warning" if lan_pred > 15 else "Normal"
        }
        
        print(f"📤 [Debug] 正在写回 LAN AI 结果...")
        firebase_db.reference("56214328/ai_analysis").update(update_data)
        print(f"✅ [Debug] LAN AI 写入成功: {lan_pred}")

    except Exception as e:
        print(f"❌ [Debug] LAN 处理器内部错误: {e}")

# ==========================================
#  4. WAN 逻辑 + 详细日志
# ==========================================
def handle_wan_data(event):
    print(f"🔔 [Event] WAN 节点更新! 路径: {event.path}")
    if event.data is None: return
    
    try:
        readings = event.data.get('readings', {})
        print(f"📥 [Debug] WAN 收到 {len(readings)} 个监测站数据")

        # 模拟/计算
        wan_pred = 4.0 # 默认值
        if wan_engine:
            vals = [float(v) for v in readings.values()]
            wan_pred = sum(vals) / len(vals)
        
        print(f"📤 [Debug] 正在写回 WAN AI 结果...")
        firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
            "territory_avg_prediction": round(float(wan_pred), 4),
            "last_calc_time": datetime.datetime.utcnow().isoformat() + "Z",
            "status": "Moderate"
        })
        print(f"✅ [Debug] WAN AI 写入成功")
    except Exception as e:
        print(f"❌ [Debug] WAN 处理器错误: {e}")

# ==========================================
#  5. 初始化与监听
# ==========================================
def start_services():
    print("🚀 [Debug] 正在启动 Firebase 服务...")
    fb_config_str = os.environ.get("FIREBASE_CONFIG")
    
    if fb_config_str:
        print("📦 [Debug] 发现环境变量 FIREBASE_CONFIG")
        cred = credentials.Certificate(json.loads(fb_config_str, strict=False))
    else:
        print("⚠️ [Debug] 未发现环境变量，尝试读取本地 json")
        cred = credentials.Certificate(os.path.join(BASE_DIR, "serviceAccountKey.json"))

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
    
    print("🔥 [Debug] Firebase SDK 初始化完成")

    # 启动监听
    try:
        print("🎧 [Debug] 正在注册 LAN 监听器: 56214328/latest")
        firebase_db.reference("56214328/latest").listen(handle_lan_data)
        
        print("🎧 [Debug] 正在注册 WAN 监听器: GAGNN_24hours/GAGNN_data")
        firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)
        
        print("📡 [Debug] 监听通道已全部开启")
    except Exception as e:
        print(f"❌ [Debug] 监听器启动失败: {e}")

@app.route('/')
def home():
    print("🌐 [Debug] 收到 Web 访问请求")
    return jsonify({"status": "Online", "debug_mode": True})

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    print(f"🏁 [Debug] Flask 即将运行在端口: {port}")
    app.run(host='0.0.0.0', port=port)
