import os
import threading
import torch
import datetime
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask

# --- 1. GNN 模型結構 (必須與訓練時一致) ---
# 根據你的 JS，輸入特徵為 5 個: [mq135.norm, mq2.norm, mq7.norm, temp, humidity]
class BuildTechGNN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1):
        super(BuildTechGNN, self).__init__()
        try:
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        except ImportError:
            # 如果環境沒裝好 torch-geometric 的備用簡單層
            self.conv1 = torch.nn.Linear(input_dim, hidden_dim)
            self.conv2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- 2. 初始化環境 ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Firebase 初始化 (使用你指定的 asia-southeast1 網址)
if not firebase_admin._apps:
    cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://project-12cc8-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

# 載入模型權重
def load_model():
    model = BuildTechGNN(input_dim=5, hidden_dim=16, output_dim=1)
    model_path = os.path.join(BASE_DIR, "model.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("✅ GNN Model Weights Loaded")
        except Exception as e:
            print(f"⚠️ Model load error: {e}")
    model.eval()
    return model

gnn_model = load_model()

# --- 3. 核心監聽與預測函數 ---
def run_gnn_prediction(event):
    """
    當 firebase.js 更新 '56214328/latest' 時，此函數會自動觸發
    """
    data = event.data
    if data is None: return

    try:
        # 1. 依照 firebase.js 結構精確提取數據
        sensors = data.get('sensors', {})
        mq135 = sensors.get('mq135', {}).get('norm', 0)
        mq2   = sensors.get('mq2', {}).get('norm', 0)
        mq7   = sensors.get('mq7', {}).get('norm', 0)

        weather = data.get('weather', {})
        temp = weather.get('temp', 0)
        hum  = weather.get('humidity', 0)

        # 2. 構建 Tensor (5 個特徵)
        # 格式: [[mq135, mq2, mq7, temp, hum]]
        x = torch.tensor([[mq135, mq2, mq7, temp, hum]], dtype=torch.float)
        
        # 3. 靜態圖結構 (單節點自環，適用於單站點監控)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # 4. 執行 AI 預測
        with torch.no_grad():
            prediction = gnn_model(x, edge_index)
            score = prediction.item()

        # 5. 寫回結果到 ai_analysis
        db.reference('56214328/ai_analysis').update({
            "current_prediction": round(score, 4),
            "status": "Warning" if score > 0.5 else "Safe", # 閾值可依需求調整
            "last_calc_time": datetime.datetime.now().isoformat()
        })
        print(f"🚀 [GNN Update] Score: {round(score,4)} | Data: T:{temp} H:{hum}")

    except Exception as e:
        print(f"❌ Prediction Logic Error: {e}")

# --- 4. 運行與監控 ---
@app.route('/')
def home():
    return "BuildTech AI Node is Live", 200

def start_listener():
    # 監聽路徑必須與 firebase.js 推送的路徑完全一致
    print("📡 Monitoring Firebase path: 56214328/latest")
    db.reference('56214328/latest').listen(run_gnn_prediction)

if __name__ == "__main__":
    # 啟動背景監聽線程
    threading.Thread(target=start_listener, daemon=True).start()
    
    # 啟動 Flask (Render 偵測 Port 10000)
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
