import os
import torch
import datetime
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask

# --- 1. 模型結構定義 (名稱必須與 model.pth 一致) ---
class BuildTechGNN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1):
        super(BuildTechGNN, self).__init__()
        # 修正：將 fc 改回 conv 以對應你的 model.pth
        self.conv1 = torch.nn.Linear(input_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 這裡也要同步修改
        x = torch.relu(self.conv1(x))
        return self.conv2(x)

# --- 2. 立即定義全域變數 ---
gnn_model = BuildTechGNN(input_dim=5)
gnn_model.eval()

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS = {"firebase": "Initializing...", "model": "Initial", "last_prediction": "None"}

# --- 3. 預測函數 ---
def run_gnn_prediction(event):
    data = event.data
    if data is None: return

    try:
        sensors = data.get('sensors', {})
        mq135 = sensors.get('mq135', {}).get('norm', 0)
        mq2   = sensors.get('mq2', {}).get('norm', 0)
        mq7   = sensors.get('mq7', {}).get('norm', 0)
        weather = data.get('weather', {})
        temp = weather.get('temp', 0)
        hum  = weather.get('humidity', 0)

        x = torch.tensor([[mq135, mq2, mq7, temp, hum]], dtype=torch.float)
        with torch.no_grad():
            score = gnn_model(x).item()

        db.reference('56214328/ai_analysis').update({
            "current_prediction": round(score, 4),
            "status": "Warning" if score > 0.5 else "Safe",
            "last_calc_time": datetime.datetime.now().isoformat(),
            "engine": "BuildTech-GNN-Final-Fixed"
        })
        
        STATUS["last_prediction"] = round(score, 4)
        print(f"✅ Prediction Successful: {score}")

    except Exception as e:
        print(f"❌ Prediction Error: {e}")

# --- 4. 初始化 ---
def start_system():
    # A. 載入模型權重
    model_path = os.path.join(BASE_DIR, "model.pth")
    if os.path.exists(model_path):
        try:
            # 現在名稱匹配了，載入應該會成功
            gnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            STATUS["model"] = "Weights Loaded Successfully"
        except Exception as e:
            STATUS["model"] = f"Weight Error: {e}"
    else:
        STATUS["model"] = "No pth file found"

    # B. Firebase
    try:
        cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        STATUS["firebase"] = "Connected"
        db.reference('56214328/latest').listen(run_gnn_prediction)
    except Exception as e:
        STATUS["firebase"] = f"Error: {e}"

@app.route('/')
def health():
    return STATUS, 200

if __name__ == "__main__":
    start_system()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
