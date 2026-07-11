
# ===== 森林空地协同巡检 Demo v1.0 =====
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import cv2, os, time, json
from IPython.display import clear_output

# 加载模型
model = YOLO("best.pt")
model.conf = 0.25

print("="*60)
print("🌲 森林空地协同巡检系统 Demo")
print("="*60)

test_dir = "test_images"
if not os.path.exists(test_dir):
    print("❌ 请将测试图片放在 test_images/ 文件夹下")
    exit()

test_files = [f for f in os.listdir(test_dir) if f.endswith((".jpg", ".png"))][:8]

detected_count = 0
for idx, fname in enumerate(test_files):
    clear_output(wait=True)
    img_path = os.path.join(test_dir, fname)
    
    print(f"\n{'='*60}")
    print(f"🛸 第{idx+1}/{len(test_files)}次巡航  |  正在扫描区域...")
    print(f"{'='*60}")
    
    time.sleep(0.3)
    
    results = model(img_path)
    boxes = results[0].boxes
    
    print(f"📷 拍摄画面：{fname}")
    cv2_imshow(results[0].plot())
    
    if len(boxes) > 0:
        detected_count += 1
        cls = int(boxes[0].cls[0])
        conf = float(boxes[0].conf[0])
        name = model.names[cls]
        
        print(f"\n⚠️ ⚠️ ⚠️  发现异常！⚠️ ⚠️ ⚠️")
        print(f"   类别：{name}")
        print(f"   置信度：{conf:.2%}")
        print(f"   坐标：东经 116.3984°  北纬 39.9512°")
        
        time.sleep(0.3)
        print(f"\n📡 正在下发任务给无人车...")
        print(f"   📋 目标：{name} | 坐标：116.3984, 39.9512")
        
        print(f"\n🚗 无人车已出发...")
        time.sleep(0.3)
        print(f"   ✅ 已到达目标点")
        print(f"   📸 近距离拍照采集...")
        print(f"   ✅ 近检完成，结果已回传")
        print(f"\n📊 巡检报告：发现 {name} ✅")
    else:
        print(f"\n✅ 未发现异常，继续巡航...")
    
    print(f"{'='*60}")
    time.sleep(0.5)

print(f"\n{'='*60}")
print(f"🏁 巡检任务完成！共检测 {len(test_files)} 个区域")
print(f"   发现异常：{detected_count} 处")
print(f"{'='*60}")
