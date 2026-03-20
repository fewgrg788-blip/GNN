import os
import threading
import torch
import datetime
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask

# --- 1. 模型結構 ---
class BuildTechGNN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1):
        super(BuildTechGNN, self).__init__()
        self.conv1 = torch.nn.Linear(input_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS = {"model": "Loading...", "firebase": "Connecting...", "last_data": "None"}

# --- 2. 核心監聽邏輯 ---
def run_gnn_prediction(event):
    """當 Firebase 數據變動時觸發"""
    print(f"🔔 [Firebase Event] 偵測到數據變動: {event.path}")
    data = event.data
    if data is None: 
        print("⚠️ 收到空數據，跳過")
        return

    try:
        # 紀錄最後收到的數據時間
        STATUS["last_data"] = datetime.datetime.now().isoformat()
        
        # 解析數據 (嚴格對應你的 JSON 結構)
        sensors = data.get('sensors', {})
        mq135 = sensors.get('mq135', {}).get('norm', 0)
        mq2   = sensors.get('mq2', {}).get('norm', 0)
        mq7   = sensors.get('mq7', {}).get('norm', 0)
        
        weather = data.get('weather', {})
        temp = weather.get('temp', 0)
        hum  = weather.get('humidity', 0)

        print(f"📊 處理中: MQ135({mq135}), MQ2({mq2}), MQ7({mq7}), T({temp}), H({hum})")

        # 轉為 Tensor 並預測
        x = torch.tensor([[mq135, mq2, mq7, temp, hum]], dtype=torch.float)
        with torch.no_grad():
            score = gnn_model(x).item()

        # 寫回 Firebase (注意路徑：我們直接寫在 56214328 下面)
        result_ref = db.reference('56214328/ai_analysis')
        result_data = {
            "current_prediction": round(score, 4),
            "status": "Warning" if score > 0.5 else "Safe",
            "last_calc_time": datetime.datetime.now().isoformat()
        }
        result_ref.update(result_data)
        print(f"✅ 預測成功並寫回 Firebase: {score}")

    except Exception as e:
        print(f"❌ 預測執行出錯: {e}")

# --- 3. 啟動與初始化 ---
def initialize_system():
    global gnn_model
    # 初始化模型
    gnn_model = BuildTechGNN(input_dim=5)
    model_path = os.path.join(BASE_DIR, "model.pth")
    if os.path.exists(model_path):
        gnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        STATUS["model"] = "Loaded"
    else:
        STATUS["model"] = "Error: model.pth missing"
    gnn_model.eval()

    # 初始化 Firebase
    try:
        cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        STATUS["firebase"] = "Connected"
        
        # ⚠️ 關鍵：開始監聽
        print("📡 開始監聽路徑: 56214328/latest")
        db.reference('56214328/latest').listen(run_gnn_prediction)
    except Exception as e:
        STATUS["firebase"] = f"Error: {e}"

@app.route('/')
def health():
    return STATUS, 200

if __name__ == "__main__":
    # 先跑初始化再開網頁
    initialize_system()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
