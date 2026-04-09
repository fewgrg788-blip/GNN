import os
import sys
import json
import torch
import datetime
import numpy as np
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify, request

# --- [1. 環境與模型架構導入] ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"📂 [System] 當前根目錄: {BASE_DIR}")

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
# 2. 站點名稱映射字典
# ==========================================
STATION_MAPPING = {
    'AQHI_Central/Western': 'Central_Western_General',
    'AQHI_Eastern': 'Eastern_General',
    'AQHI_Kwun Tong': 'Kwun_Tong_General',
    'AQHI_Sham Shui Po': 'Sham_Shui_Po_General',
    'AQHI_Kwai Chung': 'Kwai_Chung_General',
    'AQHI_Tsuen Wan': 'Tsuen_Wan_General',
    'AQHI_Tseung Kwan O': 'Tseung_Kwan_O_General',
    'AQHI_Yuen Long': 'Yuen_Long_General',
    'AQHI_Tuen Mun': 'Tuen_Mun_General',
    'AQHI_Tung Chung': 'Tung_Chung_General',
    'AQHI_Tai Po': 'Tai_Po_General',
    'AQHI_Sha Tin': 'Sha_Tin_General',
    'AQHI_North': 'North_General',
    'AQHI_Tap Mun': 'Tap_Mun_General',
    'AQHI_Causeway Bay': 'Causeway_Bay_Roadside',
    'AQHI_Central': 'Central_Roadside',
    'AQHI_Mong Kok': 'Mong_Kok_Roadside',
    'AQHI_Southern': 'Southern_General'
}

# ==========================================
# 3. AI 引擎初始化
# ==========================================
def init_ai_engine():
    global wan_model, lan_model, adj_data, STATION_ORDER
    try:
        print("🛠️ [Init] 正在加載 AI 核心組件...")
        
        adj_path = os.path.join(BASE_DIR, "models", "adjacency_pyg.pt")
        if not os.path.exists(adj_path):
            print(f"❌ [Critical] 找不到圖結構文件: {adj_path}")
            return
            
        adj_data = torch.load(adj_path, map_location=DEVICE)
        STATION_ORDER = adj_data['stations']
        print(f"📡 [Graph] 地理矩陣加載成功 | 站點數量: {len(STATION_ORDER)}")

        wan_model = HK_Pro_Model(node_features=3, hidden_dim=128, seq_len=24, horizon=6)
        wan_weight = os.path.join(BASE_DIR, "weights", "hk_pro_model_best.pth")
        if os.path.exists(wan_weight):
            wan_model.load_state_dict(torch.load(wan_weight, map_location=DEVICE))
            print(f"📦 [WAN Model] 成功加載 Pro 權重")
        
        wan_model.eval()

        lan_model = BuildTechGNN()
        lan_weight = os.path.join(BASE_DIR, "weights", "model_lan.pth")
        if os.path.exists(lan_weight):
            lan_model.load_state_dict(torch.load(lan_weight, map_location=DEVICE))
            print(f"📦 [LAN Model] 成功加載實時模型權重")
        lan_model.eval()

        print("🚀 [System] 雙模 AI 引擎啟動成功")
    except Exception as e:
        print(f"❌ [System] 初始化崩潰: {e}")

# ==========================================
# 4. 路由與接口定義 (Flask Routes)
# ==========================================

@app.route('/')
def health_check():
    return jsonify({
        "status": "online",
        "engine": "GAGNN-Pro-Hybrid",
        "last_ping": datetime.datetime.now(datetime.UTC).isoformat() + "Z"
    })

@app.route('/predict', methods=['POST']) # 必須是 /predict
def predict():
    """供 GitHub Action 主動調用的預測接口"""
    if wan_model is None:
        return jsonify({"error": "AI Model not initialized"}), 500
    
    try:
        req_data = request.get_json()
        input_list = req_data.get('data', [])
        
        if len(input_list) != 18:
            return jsonify({"error": f"Expected 18 stations, got {len(input_list)}"}), 400

        # 推理邏輯
        curr_vals = [float(v) / 10.0 for v in input_list]
        history_buffer = np.tile(curr_vals, (24, 1)) 
        input_tensor = torch.zeros((1, 18, 24, 3))
        input_tensor[0, :, :, 0] = torch.FloatTensor(history_buffer).T
        
        with torch.no_grad():
            prediction = wan_model(input_tensor, adj_data['edge_index']).numpy()[0]
        
        # 獲取 T+6 (第 6 個步長) 的預測
        pred_t6 = (prediction[:, 5] * 10.0).tolist()
        
        print(f"✅ [API] GitHub Action 請求成功 | T+6 平均: {np.mean(pred_t6):.2f}")
        return jsonify({
            "status": "success",
            "prediction": [round(v, 2) for v in pred_t6],
            "target": "T+6 Hours"
        })

    except Exception as e:
        print(f"❌ [API Error] {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 5. Firebase 監聽邏輯 (被動觸發)
# ==========================================
def handle_wan_data(event):
    if event.data is None or wan_model is None: return

    try:
        readings = event.data.get('readings', {})
        if not readings: return

        curr_vals = []
        for s in STATION_ORDER:
            firebase_key = STATION_MAPPING.get(s, s) 
            val = float(readings.get(firebase_key, 0))
            curr_vals.append(val / 10.0)
        
        history_buffer = np.tile(curr_vals, (24, 1)) 
        input_tensor = torch.zeros((1, 18, 24, 3))
        input_tensor[0, :, :, 0] = torch.FloatTensor(history_buffer).T
        
        with torch.no_grad():
            prediction = wan_model(input_tensor, adj_data['edge_index']).numpy()[0]
        
        future_val = np.mean(prediction[:, 5]) * 10.0
        
        # 更新 Firebase
        firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
            "territory_avg_6h": round(float(future_val), 2),
            "last_update": datetime.datetime.now(datetime.UTC).isoformat() + "Z"
        })
        print(f"📡 [Listener] Firebase 數據已同步更新")

    except Exception as e:
        print(f"❌ [WAN Listener] 出錯: {e}")

# ==========================================
# 6. 啟動入口
# ==========================================
def start_services():
    print("🌐 [System] 正在連接 Firebase...")
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
    
    init_ai_engine()
    
    # 開啟實時監聽
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    print(f"📡 [Flask] API 服務正在啟動於 Port {port}...")
    app.run(host='0.0.0.0', port=10000)
