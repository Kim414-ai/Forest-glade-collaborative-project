# ===== 本地版 Demo =====
from ultralytics import YOLO
import cv2, os

model = YOLO("best.pt")
model.conf = 0.01

print("="*50)
print("森林空地协同巡检系统 Demo")
print("="*50)

test_dir = "test_images"
if not os.path.exists(test_dir):
    print("请新建 test_images 文件夹，放几张烟雾图片进去")
    exit()

test_files = [f for f in os.listdir(test_dir) if f.endswith((".jpg",".png"))]

detected = 0
for idx, fname in enumerate(test_files):
    img_path = os.path.join(test_dir, fname)
    print(f"\n第{idx+1}/{len(test_files)}次巡航...")
    
    results = model(img_path)
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        detected += 1
        cls = int(boxes[0].cls[0])
        conf = float(boxes[0].conf[0])
        print(f"发现异常：{model.names[cls]}，置信度 {conf:.2%}")
        print("下发任务给无人车...")
        print("无人车已到达，近检完成")
    else:
        print("未发现异常")
    
    cv2.imshow("Patrol Result", results[0].plot())
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\n完成！发现异常 {detected} 处")