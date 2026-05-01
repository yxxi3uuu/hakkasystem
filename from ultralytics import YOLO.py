from ultralytics import YOLO

# 1. 載入模型
model = YOLO("yolo11n.pt")

# 2. 進行偵測 (請注意路徑中的斜線建議改為正斜線 / 或是加 r 防止轉義)
results = model.predict(source=r"C:\Users\user\Downloads\下載.jpg")

# 3. 顯示結果
results[0].show()