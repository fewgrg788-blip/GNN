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

try:
    from models.model_pro import HK_Pro_Model
    from models.lan_gnn import BuildTechGNN
    print("✅ [AI Engine] 成功偵測到 Pro 與 LAN 模型架構定義")
except ImportError as e:
    print(f"❌ [Error] 導入失敗: {e}. 請檢查 models/ 是否包含 __init__.py 和相關模型文件")

app = Flask(__name__)
DEVICE = torch.device("cpu")

# 全域變數
wan_model = None
lan_model = None
adj_data = None
STATION_ORDER = []

# ==========================================
# 2. 站點名稱映射字典 (關鍵修復：解決 0.0 問題)
# ==========================================
# 將 Firebase 的 key 對應到 adjacency_pyg.pt 的站點名稱邏輯
# 如果未來名稱有變，只需在這裡修改對應關係
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
# 4. 預測邏輯 (WAN) 與 雲端反饋
# ==========================================
def handle_wan_data(event):
    if event.data is None: return
    if wan_model is None:
        print("❌ [WAN] 模型未就緒，無法執行預測")
        return

    try:
        print(f"\n🔍 [WAN] 偵測到數據變動...")
        readings = event.data.get('readings', {})
        
        if not readings:
            print("⚠️ [WAN] 數據中不包含 readings 欄位，略過分析")
            return

        # 1. 提取並使用字典映射數據
        curr_vals = []
        debug_vals = []
        for s in STATION_ORDER:
            # 使用字典尋找對應的 Firebase 鍵值
            firebase_key = STATION_MAPPING.get(s, s) 
            
            # 獲取數值，如果找不到預設為 0
            val_str = readings.get(firebase_key, 0)
            val = float(val_str)
            
            curr_vals.append(val / 10.0) # 歸一化 (0-1)
            debug_vals.append(f"{firebase_key}:{val}")
        
        print(f"📊 [WAN] 成功提取 18 區實時數據 (前3筆): {debug_vals[:3]}")

        # 2. 構造時間序列張量 [1, 18, 24, 3]
        history_buffer = np.tile(curr_vals, (24, 1)) 
        input_tensor = torch.zeros((1, 18, 24, 3))
        input_tensor[0, :, :, 0] = torch.FloatTensor(history_buffer).T
        
        # 3. 執行 GNN 推理
        print("🧠 [WAN] 正在執行時空預測推理 (GAT + GRU)...")
        with torch.no_grad():
            prediction = wan_model(input_tensor, adj_data['edge_index']).numpy()[0]
        
        # 4. 結果計算 (T+6 小時全港平均)
        future_val = np.mean(prediction[:, 5]) * 10.0
        current_avg = np.mean(curr_vals) * 10.0
        print(f"🔮 [WAN] 推理完成 | 當前平均: {current_avg:.2f} -> 6小時後預估: {future_val:.2f}")

        # 5. 更新雲端結果
        status_str = "Hazardous" if future_val > 7.0 else "Safe"
        firebase_db.reference("GAGNN_24hours/wan_ai_analysis").update({
            "territory_avg_current": round(float(current_avg), 2),
            "territory_avg_6h": round(float(future_val), 2),
            "engine": "GAGNN-Pro-Hybrid-Engine",
            "last_update": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            "status": status_str
        })

        # ==========================================
        # 🟢 反饋邏輯 (Feedback Loop) -> 指揮 ESP32
        # ==========================================
        feedback_ref = firebase_db.reference("56214328/config") 
        
        if future_val > 7.0:
            # 情況 A：預測未來污染嚴重 -> 要求 ESP32 加密監控
            feedback_ref.update({
                "sample_rate_min": 10,     
                "alert_mode": True,        
                "system_msg": "AI_PREDICT_HIGH_RISK"
            })
            print(f"⚠️ [Feedback] 警告！未來風險高，已下發「加密採集(10min)」指令至設備")
        elif current_avg > 5.0 and future_val > current_avg:
             # 情況 B：污染正在快速惡化 -> 提前進入警戒
            feedback_ref.update({
                "sample_rate_min": 30,     
                "alert_mode": False,        
                "system_msg": "AI_PREDICT_DETERIORATING"
            })
            print(f"📈 [Feedback] 污染趨勢上升中，已下發「警戒模式(30min)」指令至設備")
        else:
            # 情況 C：未來環境安全
            feedback_ref.update({
                "sample_rate_min": 60,     
                "alert_mode": False,
                "system_msg": "AI_PREDICT_NORMAL"
            })
            print(f"✅ [Feedback] 預期環境穩定，維持標準低功耗模式(60min)")
        print("-" * 40)

    except Exception as e:
        print(f"❌ [WAN] 運行時出錯: {e}")

# ==========================================
# 5. 啟動服務
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
    
    print("👂 [Listener] 正在開啟 Firebase 即時監聽隊列...")
    firebase_db.reference("GAGNN_24hours/GAGNN_data").listen(handle_wan_data)

@app.route('/')
def health_check():
    return jsonify({
        "status": "online",
        "engine": "GAGNN-Pro-Hybrid",
        "device_target": "56214328",
        "last_ping": datetime.datetime.now(datetime.UTC).isoformat() + "Z"
    })

if __name__ == "__main__":
    start_services()
    port = int(os.environ.get("PORT", 10000))
    print(f"📡 [Flask] API 服務正在啟動於 Port {port}...")
    app.run(host='0.0.0.0', port=port)
