from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model.model import TransformerModel
from sklearn.preprocessing import MinMaxScaler
import logging
from fastapi.middleware.cors import CORSMiddleware

# uvicorn main:app --reload
# uvicorn main:app --reload --port 8080
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=False,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.disabled = True

# Pydantic 모델 정의
class InputData(BaseModel):
    lot_id: str
    temperature: float
    current: float

g_model = TransformerModel(seq_length=18, input_dim=2, d_model=36, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1)
g_model.load_state_dict(torch.load('./model/transformer_model.pth'))
g_model.eval()

@app.post("/predict")
async def predict(data: InputData):
    try:
        model = g_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print("input data:", data)


        # 정상 온도와 전류
        normal_temp = 70.93422226600362
        normal_current = 1.6057822320811486
        threshold = 0.5

        # 스케일러 설정 및 정상 값 스케일링
        scaler = MinMaxScaler()
        scaler.fit([[normal_temp, normal_current]])  # 전체 데이터셋의 통계를 이용해 스케일러 훈련
        scaled_normal = scaler.transform([[normal_temp, normal_current]])

        # 입력 데이터 스케일링
        input_data = np.array([[data.temperature, data.current]] * 18)  # seq_length 만큼 반복
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).view(1, -1).to(device)  # (1, 36) 차원으로 변환

        with torch.no_grad():
            output = model(input_tensor, input_tensor)
            output = output.view(1, -1)  # 출력도 적절한 형태로 조정
            mse_loss = F.mse_loss(output, input_tensor, reduction='none')
            mse = mse_loss.mean().item()

        is_abnormal = int(mse > threshold)
        mse_temp = mse_loss[..., 0::2].mean().item()  # 모든 온도 기여도 계산
        mse_current = mse_loss[..., 1::2].mean().item()  # 모든 전류 기여도 계산
        total_mse = mse_temp + mse_current
        temp_contribution = round(mse_temp / total_mse, 1) if total_mse != 0 else 0
        current_contribution = round(mse_current / total_mse, 1) if total_mse != 0 else 0
        print("total MSE:", total_mse, "temperature MSE:", mse_temp, "current MSE:", mse_current)
        print("temperature contribution:", temp_contribution, "current contribution:", current_contribution)

        # 스케일링된 입력 값과 정상 값 비교
        temp_tendency = 1 if input_scaled[0, 0] > scaled_normal[0, 0] else 0
        current_tendency = 1 if input_scaled[0, 1] > scaled_normal[0, 1] else 0
        print("temperature tendency:", temp_tendency, "current tendency:", current_tendency)
        print("normal type:", is_abnormal)

        return {
            'lot_id': data.lot_id,
            'normal_type': is_abnormal,
            'temperature_tendency': temp_tendency,
            'current_tendency': current_tendency,
            'temperature_contribution': temp_contribution,
            'current_contribution': current_contribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
