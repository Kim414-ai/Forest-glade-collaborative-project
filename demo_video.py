# ===== 森林巡检 Demo（视频版 + 连续三帧告警）=====
from ultralytics import YOLO
import cv2, os

# 加载新模型
model = YOLO("best_fire_smoke.pt")
model.conf = 0.25

print("="*50)
print("森林空地协同巡检系统 Demo（视频版）")
print("="*50)
print(f"模型类别：{model.names}")
print("="*50)

# 视频文件（你放一段森林/火灾视频，改这里）
video_path = "test_video.mp4"
if not os.path.exists(video_path):
    print(f"❌ 找不到视频 {video_path}，请放一段视频进来并改脚本里的文件名")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 视频打不开（编码问题，转成H.264的mp4）")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# FPS修正（有些视频读出来是天文数字）
if fps > 120:
    fps = 25

print(f"视频：{w}x{h}, {fps}fps, 共{total}帧")

# 输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("detect_result.mp4", fourcc, fps, (w, h))

consecutive = 0          # 连续检测到异常的次数
MIN_FRAMES = 3           # 连续几帧才告警
alarm_frames = 0         # 累计告警次数
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 每隔2帧检测一次（加速，跳过帧）
    frame_count += 1
    if frame_count % 2 != 0:
        out.write(frame)
        continue

    results = model(frame)
    boxes = results[0].boxes

    # 找有没有 fire / smoke
    abnormal = None
    for box in boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if name in ["fire", "smoke"]:
            abnormal = (name, conf)
            break

    # 连续帧计数
    if abnormal:
        consecutive += 1
    else:
        consecutive = 0

    annotated = results[0].plot()

    # 达到连续阈值 → 告警
    if consecutive >= MIN_FRAMES:
        alarm_frames += 1
        cv2.putText(annotated, "ALARM!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.rectangle(annotated, (0, 0), (w-1, h-1), (0, 0, 255), 5)
        print(f"🚨 第{frame_count}帧：连续{consecutive}帧检测到 {abnormal[0]}，触发告警！")

    # 状态显示
    cv2.putText(annotated, f"consecutive: {consecutive}", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(annotated)

    # 实时预览（按q退出）
    cv2.imshow("Patrol", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("="*50)
print(f"完成！共处理 {frame_count} 帧，触发告警 {alarm_frames} 次")
print("结果视频已保存为 detect_result.mp4")