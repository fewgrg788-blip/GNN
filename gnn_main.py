import os
import json
import firebase_admin
from firebase_admin import credentials

# 导入你的 GNN 模型 (确保文件夹已更名为 models)
try:
    from models.lan_gnn import BuildTechGNN
    from models.wan_gnn import WAN_GNN
    print("✅ [System] Models imported successfully")
except ImportError as e:
    print(f"❌ [System] Import Error: {e}")

def start_firebase():
    try:
        # 1. 检查环境变量是否存在
        fb_config_str = os.environ.get("FIREBASE_CONFIG")
        
        if fb_config_str:
            print("📦 [System] Using Render Environment Variable for Firebase")
            # 将字符串解析为 JSON 对象
            fb_config_dict = json.loads(fb_config_str)
            cred = credentials.Certificate(fb_config_dict)
        else:
            print("📄 [System] Environment variable not found, using local JSON file")
            # 本地运行时使用文件路径
            cred = credentials.Certificate("serviceAccountKey.json")

        # 2. 初始化 Firebase (防止重复初始化)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        print("🔥 [System] Firebase Connected Successfully")
        
    except Exception as e:
        print(f"❌ [System] Firebase Connection Error: {e}")

if __name__ == "__main__":
    start_firebase()
    # 后面接你的 Flask 运行代码...
