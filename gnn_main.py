import os
import sys
import json
import torch
import datetime
import numpy as np
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify

# --- [1. 環境與模型架構導入] ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"📂 [System] 當前根目錄: {BASE_DIR}")

# 導入架構
try:
    from models.model_pro import HK_Pro_Model
    from models.lan_gnn import BuildTechGNN
    print("✅ [AI Engine] 成功偵測到 Pro 與 LAN 模型架構定義")
except ImportError as e:
    print(f"❌ [Error] 導入失敗: {e}. 請檢查 models/ 是否包含 __init__.py")

app = Flask(__name__)
DEVICE = torch.device("cpu")

# 全域變數
wan_model = None
lan_model = None
adj_data = None
STATION_ORDER = []

# ==========================================
# 2. AI 引擎初始化 (加載模型與權重)
# ==========================================
def init_ai_engine():
    global wan_model, lan_model, adj_data, STATION_ORDER
    try:
        print("🛠️ [Init] 正在加載 AI 核心組件...")
        
        # A. 加載圖結構
        adj_path = os.path.join(BASE_DIR, "models", "adjacency_pyg.pt")
        if not os.path.exists(adj_path):
            print(f"❌ [Critical] 找不到圖結構文件: {adj_path}")
            return
            
        adj_data = torch.load(adj_path, map_location=DEVICE)
        STATION_ORDER = adj_data['stations']
        print(f"📡 [Graph] 地理矩陣加載成功 | 站點數量: {len(STATION_ORDER)}")

        # B. 加載 WAN Pro 模型
        wan_model = HK_Pro_Model(node_features=3, hidden_dim=128, seq_len=24, horizon=6)
        wan_weight = os.path.join(BASE_DIR, "weights", "hk_pro_model_best.pth")
        if os.path.exists(wan_weight):
            wan_model.load_state_dict(torch.load(wan_weight, map_location=DEVICE))
            print(f"📦 [WAN Model] 成功加載 Pro 權重: {wan_weight}")
        else:
            print(f"⚠️ [WAN Model] 找不到權重文件，將使用隨機初始化模型")
        wan_model.eval()

        # C. 加載 LAN 模型
        lan_model = BuildTechGNN()
        lan_weight = os.path.join(BASE_DIR, "weights", "model_lan.pth")
        if os.path.exists(lan_weight):
            lan_model.load_state_dict(torch.load(lan_weight, map_location=DEVICE))
            print(f"📦 [LAN Model] 成功加載實時模型權重: {lan_weight}")
        lan_model.eval()

        print("🚀 [System] 雙模 AI 引擎啟動：WAN (Pro 預測) + LAN (實時分析)")
    except Exception as e:
        print(f"❌ [System] 初始化崩潰: {e}")

# ==========================================
# 3. 預測邏輯 (WAN) 與 反饋迴路
# ==========================================
def handle_wan_data(event):
    if event.data is None:
        print("ℹ️ [WAN] 收到空數據更新，忽略處理")
        return
    if wan_model is None:
        print("❌ [WAN] 模型未就緒，無法執行預測")
        return

    try:
        print(f"🔍 [WAN] 偵測到數據變動，路徑: {event.path}")
        readings = event.data.get('readings', {})
        
        if not readings:
            print("⚠️ [WAN] 數據中不包含 readings 欄位")
            return

        # 1. 提取並歸一化 (0-10 轉 0-1)
        curr_vals = []
        for s in STATION_ORDER:
            val = float(readings.get(s, 0))
            curr_vals.append(val / 10.0)
        
        print(f"📊 [WAN] 已提取 18 區當前數值 (前三名: {STATION_ORDER[:3]} -> {curr_vals[:3]})")

        # 2. 構造時間序列張量 [1, 18, 24, 3]
        # (這裡假設構造 24 小時歷史緩存，F=0 是 AQHI)
        history_buffer = np.tile(curr_vals, (24, 1)) 
        input_tensor = torch.zeros((1, 18, 24, 3))
        input_tensor[0, :, :, 0] = torch.FloatTensor(history_buffer).T
        
        # 3. 執行 GNN 推理
        print("🧠 [WAN] 正在執行時空預測推理...")
        with torch.no_grad():
            prediction = wan_model(input_tensor, adj_data['edge_index']).numpy()[0]
        
        # 4. 結果計算 (T+6 小時全港平均)
        future_val = np.mean(prediction[:, 5]) * 10.0
        print(f"🔮 [WAN] 預測完成：6 小時後全港 AQHI 預估為 {future_val:.2f}")

        # 5. 更新雲端結果
        status_str = "Hazardous" if future_val > 7.0 else "Safe"
        firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
            "territory_avg_6h": round(float(future_val), 2),
            "engine": "GAGNN-Pro-V2-Engine",
            "last_update": datetime.datetime.now().isoformat(),
            "status": status_str
        })

        # ==========================================
        # 🟢 新增：反饋邏輯 (Feedback Logic)
        # 雲端大腦 (WAN) 指揮 硬體終端 (LAN/ESP32)
        # ==========================================
        feedback_ref = firebase_db.reference("56214328/config") # 假設 56214328 是您的設備 ID
        
        if future_val > 7.0:
            # 情況 A：預測未來污染嚴重
            feedback_ref.update({
                "sample_rate_min": 10,     # 加密採集頻率 (從 60min 變 10min)
                "alert_mode": True,        # 開啟強迫警報模式
                "system_msg": "AI_PREDICT_HIGH_RISK"
            })
            print(f"⚠️ [Feedback] 警告！未來風險高，已下發「加密採集」指令至設備")
        else:
            # 情況 B：未來環境安全
            feedback_ref.update({
                "sample_rate_min": 60,     # 恢復正常頻率
                "alert_mode": False,
                "system_msg": "AI_PREDICT_NORMAL"
            })
            print(f"✅ [Feedback] 未來環境穩定，設備維持低功耗模式")

    except Exception as e:
        print(f"❌ [WAN] 運行時出錯: {e}")

# ==========================================
# 4. 啟動服務
# ==========================================
def start_services():
    print("🌐 [System] 正在連接 Firebase 服務...")
    fb_config = os.environ.get("FIREBASE_CONFIG")
    if fb_config:
        cred = credentials.Certificate(json.loads(fb_config))
    else:
        cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        cred = credentials.Certificate(cred_path)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
    print("🔗 [Firebase] 連接已建立")
    
    init_ai_engine()
    
    # 啟動監聽器
    print("👂 [Listener] 正在開啟 Firebase 即時監聽隊列...")
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)

@app.route('/')
def health_check():
    return jsonify({
        "status": "online",
        "engine": "GAGNN-Pro-Hybrid",
        "device_target": "56214328",
        "last_ping": datetime.datetime.now().isoformat()
    })

if __name__ == "__main__":
    start_services()
    # Render 部署使用端口 10000
    port = int(os.environ.get("PORT", 10000))
    print(f"📡 [Flask] API 服務正在啟動於 Port {port}...")
    app.run(host='0.0.0.0', port=port)
