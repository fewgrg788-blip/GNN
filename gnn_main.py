import os
import sys
import json
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from flask import Flask, jsonify

# --- 1. 核心路徑修正 (解決 Render 找不到 models 導出問題) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- 2. 導入你的 GNN 模型 ---
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    print("✅ [System] Models imported successfully")
except ImportError as e:
    print(f"❌ [System] Import Error: {e}")
    print(f"🔍 [Debug] Current Sys Path: {sys.path}")

app = Flask(__name__)

# --- 3. 初始化 Firebase (優先使用環境變量) ---
def start_firebase():
    try:
        fb_config_str = os.environ.get("FIREBASE_CONFIG")
        
        if fb_config_str:
            print("📦 [System] Using Render Environment Variable for Firebase")
            # 使用 strict=False 增加 JSON 解析的容錯率
            fb_config_dict = json.loads(fb_config_str, strict=False)
            cred = credentials.Certificate(fb_config_dict)
        else:
            print("📄 [System] Environment variable not found, using local file")
            cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
            cred = credentials.Certificate(cred_path)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        print("🔥 [System] Firebase Connected Successfully")
        return True
    except Exception as e:
        print(f"❌ [System] Firebase Connection Error: {e}")
        return False

# --- 4. 簡單的 API 路由 (供 Render 存活檢查使用) ---
@app.route('/')
def home():
    try:
        # 測試讀取一次 Firebase 數據 (56214328 節點)
        ref = firebase_db.reference("56214328/latest")
        data = ref.get()
        return jsonify({
            "status": "online",
            "project": "BuildTech GNN Engine",
            "firebase_connected": True,
            "latest_data": data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- 5. 啟動程序 ---
if __name__ == "__main__":
    # 先啟動 Firebase
    connected = start_firebase()
    
    # 獲取 Render 分配的端口 (默認 10000)
    port = int(os.environ.get("PORT", 10000))
    
    if connected:
        print(f"🚀 [BuildTech] GNN Engine starting on port {port}...")
        # 必須使用 0.0.0.0 才能讓外部訪問
        app.run(host='0.0.0.0', port=port)
    else:
        print("❌ [Critical] Failed to connect to Firebase. System halted.")
