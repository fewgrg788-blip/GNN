import os
import sys
import json
import firebase_admin
from firebase_admin import credentials

# --- 核心修改 1: 强制路径修正 ---
# 这确保了无论从哪个目录启动，Python 都能找到同级的 models 文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- 核心修改 2: 延迟导入模型 ---
# 在路径设置好之后再导入，防止 ImportError
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    print("✅ [System] Models imported successfully")
except ImportError as e:
    print(f"❌ [System] Import Error: {e}")
    print(f"🔍 [Debug] Current Sys Path: {sys.path}")

def start_firebase():
    try:
        fb_config_str = os.environ.get("FIREBASE_CONFIG")
        
        if fb_config_str:
            print("📦 [System] Using Render Environment Variable for Firebase")
            # 解决可能的换行符转义问题
            fb_config_dict = json.loads(fb_config_str, strict=False)
            cred = credentials.Certificate(fb_config_dict)
        else:
            print("📄 [System] Environment variable not found, trying local file")
            # 本地测试路径
            cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
            cred = credentials.Certificate(cred_path)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        print("🔥 [System] Firebase Connected Successfully")
        
    except Exception as e:
        print(f"❌ [System] Firebase Connection Error: {e}")

# --- 核心修改 3: 确保 Flask 绑定正确端口 ---
if __name__ == "__main__":
    start_firebase()
    
    # 如果你使用 Flask，请确保这样写：
    # from flask_app import app # 假设你的 Flask app 在这里
    # port = int(os.environ.get("PORT", 10000))
    # app.run(host='0.0.0.0', port=port)
