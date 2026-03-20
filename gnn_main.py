import os
import threading
import torch
import datetime
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask

# --- 1. 定義模型結構 (5 維輸入) ---
class BuildTechGNN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1):
        super(BuildTechGNN, self).__init__()
        # 使用 Linear 確保即使沒裝 torch-geometric 也能啟動進行測試
        self.conv1 = torch.nn.Linear(input_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index=None):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

# --- 2. 初始化環境變數 ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS = {"model": "Loading...", "firebase": "Connecting...", "error": "None"}

# --- 3. 核心函數：載入模型與 Firebase ---
def initialize_system():
    global gnn_model
    # A. 載入 Firebase
    try:
        cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        if os.path.exists(cred_path):
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
                })
            STATUS["firebase"] = "Connected"
            # 啟動監聽器
            db.reference('56214328/latest').listen(run_gnn_prediction)
        else:
            STATUS["firebase"] = "Error: serviceAccountKey.json not found"
    except Exception as e:
        STATUS["firebase"] = f"Error: {str(e)}"

    # B. 載入模型
    gnn_model = BuildTechGNN(input_dim=5)
    try:
        model_path = os.path.join(BASE_DIR, "model.pth")
        if os.path.exists(model_path):
            gnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            gnn_model.eval()
            STATUS["model"] = "Loaded"
        else:
            STATUS["model"] = "Error: model.pth not found"
    except Exception as e:
        STATUS["model"] = f"Error: {str(e)}"

# --- 4. GNN 預測邏輯 ---
def run_gnn_prediction(event):
    data = event.data
    if data is None or gnn_model is None: return

    try:
        # 根據你的 firebase.js 結構提取
        sensors = data.get('sensors', {})
        mq135 = sensors.get('mq135', {}).get('norm', 0)
        mq2   = sensors.get('mq2', {}).get('norm', 0)
        mq7   = sensors.get('mq7', {}).get('norm', 0)

        weather = data.get('weather', {})
        temp = weather.get('temp', 0)
        hum  = weather.get('humidity', 0)

        # 轉換為 5 維 Tensor
        x = torch.tensor([[mq135, mq2, mq7, temp, hum]], dtype=torch.float)
        
        with torch.no_grad():
            prediction = gnn_model(x)
            score = prediction.item()

        # 寫回 Firebase
        db.reference('56214328/ai_analysis').update({
            "current_prediction": round(score, 4),
            "status": "Safe" if score < 0.5 else "Warning",
            "last_calc_time": datetime.datetime.now().isoformat()
        })
        print(f"Prediction Updated: {score}")
    except Exception as e:
        print(f"Prediction Error: {e}")

# --- 5. Flask 路由 (讓 Render 偵測到服務) ---
@app.route('/')
def health():
    return {
        "service": "BuildTech GNN Node",
        "firebase_status": STATUS["firebase"],
        "model_status": STATUS["model"],
        "timestamp": datetime.datetime.now().isoformat()
    }, 200

# --- 6. 啟動入口 ---
if __name__ == "__main__":
    # 使用 Thread 初始化，避免阻塞 Flask 啟動
    init_thread = threading.Thread(target=initialize_system)
    init_thread.start()

    # 立即啟動 Flask
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Web Server on port {port}...")
    app.run(host='0.0.0.0', port=port)
