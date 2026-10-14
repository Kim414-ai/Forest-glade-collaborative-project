# ===== 本地版 Demo（新模型：fire + smoke）=====
from ultralytics import YOLO
import cv2, os

# 改成新模型
model = YOLO("best_fire_smoke.pt")
model.conf = 0.25

print("="*50)
print("森林空地协同巡检系统 Demo")
print("="*50)
print(f"模型类别：{model.names}")
print("="*50)

test_dir = "test_images"
if not os.path.exists(test_dir):
    print("请新建 test_images 文件夹，放几张图片进去")
    exit()

test_files = [f for f in os.listdir(test_dir) if f.endswith((".jpg",".png"))]

detected = 0
for idx, fname in enumerate(test_files):
    img_path = os.path.join(test_dir, fname)
    print(f"\n第{idx+1}/{len(test_files)}次巡航...")

    results = model(img_path)
    boxes = results[0].boxes

    # 找有没有 fire 或 smoke（other 不算异常）
    abnormal = None
    for box in boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if cls_name in ["fire", "smoke"]:
            abnormal = (cls_name, conf)
            break

    if abnormal:
        detected += 1
        cls_name, conf = abnormal
        print(f"⚠️ 发现异常：{cls_name}，置信度 {conf:.2%}")
        print("📡 下发任务给无人车...")
        print("🚗 无人车已到达，近检完成")
    else:
        print("✅ 未发现异常")

    cv2.imshow("Patrol Result", results[0].plot())
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\n完成！发现异常 {detected} 处")