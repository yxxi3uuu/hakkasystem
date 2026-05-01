from ultralytics import YOLO

# 1. 載入模型
model = YOLO("yolo11n.pt")
img_path = r"C:\Users\ASUS\Downloads\S__27123715.jpg"
# 2. 進行偵測（不自動彈出視窗）
model = YOLO("yolo11n.pt")
results = model.predict(source=img_path, verbose=False)

# 初始化變數來記錄最高信心值
max_conf = -1.0
best_label = "未偵測到任何物體"

for result in results:
    for box in result.boxes:
        conf = float(box.conf[0])
        # 如果當前的信心值比之前紀錄的還高，就更新它
        if conf > max_conf:
            max_conf = conf
            class_id = int(box.cls[0])
            best_label = result.names[class_id]

# 最後只輸出最高的那一個
print(best_label)
