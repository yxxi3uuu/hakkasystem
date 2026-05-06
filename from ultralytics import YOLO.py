from ultralytics import YOLOWorld


model = YOLOWorld("yolov8x-worldv2.pt")
img_path = r"C:\Users\user\Downloads\S__27533329.jpg"

# 2. 進行偵測
results = model.predict(source=img_path, verbose=False)

# 初始化變數
detected_objects = []  # 存放所有物件
max_conf = -1.0        # 紀錄最高信心值
best_label = "未偵測到任何物體"

# 3. 解析預測結果
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = result.names[class_id]
        conf = float(box.conf[0])
        
        # A. 收集所有偵測到的物件
        detected_objects.append(f"{label} ({conf:.2f})")
        
        # B. 篩選最高信心值的物件 (Best Label)
        if conf > max_conf:
            max_conf = conf
            best_label = label

# 4. 輸出最終結果
if detected_objects:
    print("--- 所有偵測物件清單 ---")
    for obj in detected_objects:
        print(f"- {obj}")
    
    print("\n--- 最終決策 (Best Label) ---")
    print(f"結果: {best_label} (最高信心值: {max_conf:.2f})")
else:
    print(best_label)
