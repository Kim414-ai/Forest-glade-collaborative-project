import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import cv2
import time
import json
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# ================= 配置区 =================
MODEL_PATH    = "best.pt"
FOCUS_CLASSES = ["smoke"]
CONF          = 0.25

HIT_NEED      = 3
COOLDOWN      = 5
HEARTBEAT_SEC = 30
SHOW_POPUP    = True

VIDEO_PATH = "forest_aerial.mp4"
SOURCES = {"摄像头": 0, "航拍视频": VIDEO_PATH}

# ================= 坐标换算 =================
AREA_W_M = 100.0
AREA_H_M = 75.0
LAT0 = 30.5000
LON0 = 114.3000

DEG_PER_M_LAT = 1.0 / 111320.0
DEG_PER_M_LON = 1.0 / (111320.0 * 0.866)

def px_to_gps(x, y, frame_w=640, frame_h=480):
    meters_right = (x / frame_w) * AREA_W_M
    meters_down  = (y / frame_h) * AREA_H_M
    lon = LON0 + meters_right * DEG_PER_M_LON
    lat = LAT0 - meters_down  * DEG_PER_M_LAT
    return lat, lon

# ================= 模型 =================
print("正在加载模型：", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("模型加载完成！类别：", model.names)

def detect(frame):
    results = model(frame, imgsz=320, conf=CONF, verbose=False)
    dets = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        dets.append({"cls":  model.names[cls_id],
                     "conf": float(box.conf[0]),
                     "bbox": [x1, y1, x2, y2]})
    return dets

def draw_boxes(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        text = f'{d["cls"]} {d["conf"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 0, 255), -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ================= 区域小地图 =================
def draw_minimap(frame, target_gps, car_gps, trail):
    W = frame.shape[1]
    MW, MH = 170, 127
    X0, Y0 = W - MW - 12, 12

    cv2.rectangle(frame, (X0, Y0), (X0 + MW, Y0 + MH), (45, 45, 45), -1)
    cv2.rectangle(frame, (X0, Y0), (X0 + MW, Y0 + MH), (180, 180, 180), 1)
    cv2.putText(frame, "AREA MAP", (X0 + 6, Y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)

    def gps2mm(lat, lon):
        mx = (lon - LON0) / (AREA_W_M * DEG_PER_M_LON)
        my = (LAT0 - lat) / (AREA_H_M * DEG_PER_M_LAT)
        return int(X0 + mx * MW), int(Y0 + my * MH)

    if len(trail) >= 2:
        pts = np.array([gps2mm(la, lo) for la, lo in trail], dtype=np.int32)
        cv2.polylines(frame, [pts], False, (255, 170, 0), 2)

    if target_gps:
        tx, ty = gps2mm(target_gps[0], target_gps[1])
        cv2.circle(frame, (tx, ty), 7, (0, 0, 255), 2)
        cv2.line(frame, (tx - 10, ty), (tx + 10, ty), (0, 0, 255), 1)
        cv2.line(frame, (tx, ty - 10), (tx, ty + 10), (0, 0, 255), 1)

    if car_gps:
        cx, cy = gps2mm(car_gps[0], car_gps[1])
        cv2.rectangle(frame, (cx - 5, cy - 5), (cx + 5, cy + 5), (255, 120, 0), -1)

# ================= 视频源 =================
cap = None

def open_source(name):
    global cap
    if cap is not None:
        cap.release()
    newcap = cv2.VideoCapture(SOURCES[name])
    if not newcap.isOpened():
        print("⚠ 打不开视频源：", name, SOURCES[name])
        return False
    if name == "摄像头":
        newcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        newcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap = newcap
    return True

# ================= 界面 =================
root = tk.Tk()
root.title("森林空地协同巡检系统 v1.4")
root.configure(bg="#eceff1")

source_var = tk.StringVar(value="摄像头")

top = tk.Frame(root, bg="#263238")
top.pack(fill="x")
tk.Label(top, text="森林空地协同巡检系统", font=("微软雅黑", 15, "bold"),
         bg="#263238", fg="white", padx=12, pady=8).pack(side="left")
tk.Label(top, text=f"模型：{MODEL_PATH}", font=("微软雅黑", 10),
         bg="#263238", fg="#b0bec5", padx=10).pack(side="left")
run_status = tk.Label(top, text="● 运行中", font=("微软雅黑", 11),
                      bg="#263238", fg="#4caf50", padx=14)
run_status.pack(side="right")

mid = tk.Frame(root, bg="#eceff1")
mid.pack(pady=8)
video_label = tk.Label(mid, bg="#cfd8dc")
video_label.grid(row=0, column=0, padx=10)

right = tk.Frame(mid, bg="#eceff1")
right.grid(row=0, column=1, sticky="n")

tk.Label(right, text="实时检测", font=("微软雅黑", 11, "bold"),
         bg="#eceff1", fg="#37474f").pack(anchor="w")
det_list = tk.Listbox(right, height=5, width=26, font=("Consolas", 10))
det_list.pack(pady=(2, 8))

stat_label = tk.Label(right, text="", font=("Consolas", 10), justify="left",
                      bg="#eceff1", fg="#37474f")
stat_label.pack(anchor="w")

tk.Label(right, text="最近目标", font=("微软雅黑", 11, "bold"),
         bg="#eceff1", fg="#37474f").pack(anchor="w", pady=(8, 2))
target_label = tk.Label(right, text="—", font=("Consolas", 10), justify="left",
                        bg="#eceff1", fg="#c62828")
target_label.pack(anchor="w")

tk.Label(right, text="任务 / 小车", font=("微软雅黑", 11, "bold"),
         bg="#eceff1", fg="#37474f").pack(anchor="w", pady=(8, 2))
task_label = tk.Label(right, text="未下发", font=("Consolas", 10), justify="left",
                      bg="#eceff1", fg="#1565c0")
task_label.pack(anchor="w")
ugv_label = tk.Label(right, text="待命", font=("Consolas", 10), justify="left",
                     bg="#eceff1", fg="#2e7d32")
ugv_label.pack(anchor="w")
pb = ttk.Progressbar(right, length=190, maximum=100)
pb.pack(anchor="w", pady=(4, 0))

# ---- 状态变量 ----
paused = False
n = 0
hit_count = 0
last_alarm = 0.0
alarm_count = 0
last_target = None
last_ugv_check = 0.0
waiting_arrival = False
last_heartbeat = time.time()
task_start_time = 0.0
car_gps = None
trail = []
alarm_latched = False        # ★ 告警锁存：同一批目标只报一次

def send_task_to_ugv(target):
    task = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "task",
        "target": "ugv",
        "cls": target["cls"],
        "geo": {"lat": round(target["lat"], 6), "lon": round(target["lon"], 6)},
    }
    line = json.dumps(task, ensure_ascii=False)
    with open("ugv_tasks.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line

def send_task():
    global waiting_arrival, task_start_time
    if last_target is None:
        add_log("⚠ 还没有目标，无法下发任务")
        return
    send_task_to_ugv(last_target)
    waiting_arrival = True
    task_start_time = time.time()
    trail.clear()
    add_log(f"📡 任务已下发 → 无人车  |  目标点 "
            f"({last_target['lat']:.5f}, {last_target['lon']:.5f})")
    add_log("🚗 无人车已出发")
    task_label.config(text=f"已下发 → ({last_target['lat']:.5f}, {last_target['lon']:.5f})")

def switch_source():
    name = source_var.get()
    if open_source(name):
        print("已切换视频源：", name, SOURCES[name])
        add_log(f"🎥 视频源已切换：{name}")

def choose_video():
    path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
    if not path:
        return
    SOURCES["航拍视频"] = path
    source_var.set("航拍视频")
    switch_source()

def toggle():
    global paused
    paused = not paused
    btn.config(text="继续" if paused else "暂停")
    run_status.config(text="|| 已暂停（画面冻结，状态仍更新）" if paused else "● 运行中",
                      fg="#ffb300" if paused else "#4caf50")

def clear_log():
    log_box.delete(0, tk.END)

btns = tk.Frame(root, bg="#eceff1")
btns.pack()

btn = tk.Button(btns, text="暂停", font=("微软雅黑", 12), width=8, command=toggle)
btn.grid(row=0, column=0, padx=5)

tk.Button(btns, text="清空日志", font=("微软雅黑", 12), width=9,
          command=clear_log).grid(row=0, column=1, padx=5)

tk.Button(btns, text="下发任务 → 小车", font=("微软雅黑", 12), width=15,
          command=send_task).grid(row=0, column=2, padx=5)

combo = ttk.Combobox(btns, textvariable=source_var, values=list(SOURCES.keys()),
                     state="readonly", width=9)
combo.grid(row=0, column=3, padx=5)
combo.bind("<<ComboboxSelected>>", lambda e: switch_source())

tk.Button(btns, text="选择视频", font=("微软雅黑", 11), width=9,
          command=choose_video).grid(row=0, column=4, padx=5)

banner = tk.Label(root, text="状态正常", font=("微软雅黑", 13, "bold"),
                  bg="#2e8b57", fg="white", width=72, pady=6)
banner.pack(pady=(8, 4))

log_box = tk.Listbox(root, height=8, width=88, font=("Consolas", 10))
log_box.pack(pady=(0, 8))

def add_log(text):
    t = time.strftime("%H:%M:%S")
    log_box.insert(tk.END, f"[{t}] {text}")
    log_box.see(tk.END)
    with open("alarm_log.txt", "a", encoding="utf-8") as f:
        full_time = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{full_time}] {text}\n")

if not open_source("摄像头"):
    print("⚠ 摄像头打不开，请点【选择视频】切到航拍模式")

add_log(f"🛸 系统启动，开始巡航监测（模型 {MODEL_PATH}）")

# ================= 主循环 =================
def update():
    global n, hit_count, last_alarm, alarm_count, last_target
    global last_ugv_check, waiting_arrival, last_heartbeat
    global car_gps, alarm_latched

    now = time.time()

    # ========== ① 小车状态（★ 不受暂停影响）==========
    if now - last_ugv_check > 0.5:
        last_ugv_check = now
        try:
            with open("ugv_status.txt", "r", encoding="utf-8") as f:
                st = json.load(f)

            prog = st.get("progress", 0)
            pb["value"] = prog
            car_gps = (st["current"]["lat"], st["current"]["lon"])
            trail.append(car_gps)
            if len(trail) > 300:
                trail.pop(0)

            if st["state"] == "going":
                ugv_label.config(text=f"前往中 {prog}%  "
                                      f"({st['current']['lat']:.5f}, {st['current']['lon']:.5f})")
            elif st["state"] == "arrived":
                ugv_label.config(text="✅ 已到达目标点")
            else:
                ugv_label.config(text="待命")

            if waiting_arrival and st["state"] == "arrived":
                add_log("✅ 无人车已到达目标点")
                add_log("📸 开始近距离核查（模拟）")
                if last_target is not None:
                    add_log(f"🔎 现场确认：{last_target['cls']}   "
                            f"位置 ({last_target['lat']:.5f}, {last_target['lon']:.5f})")
                add_log(f"📊 本次任务完成，耗时 {now - task_start_time:.1f} 秒   "
                        f"|   累计告警 {alarm_count} 次")
                waiting_arrival = False

        except FileNotFoundError:
            pass
        except Exception as e:
            print("读小车状态出错：", e)

    # ========== ② 画面 + 检测（暂停时跳过）==========
    if not paused and cap is not None:
        ok, frame = cap.read()
        if not ok and source_var.get() != "摄像头":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()

        if ok:
            frame = cv2.resize(frame, (640, 480))
            n += 1
            dets = detect(frame)
            draw_boxes(frame, dets)

            det_list.delete(0, tk.END)
            for d in dets:
                det_list.insert(tk.END, f'{d["cls"]:<10} {d["conf"]:.2f}')

            targets = [d for d in dets if d["cls"] in FOCUS_CLASSES]
            if targets:
                hit_count += 1
            else:
                hit_count = 0
                alarm_latched = False            # ★ 目标消失 → 解锁

            # 告警（★ 锁存：同一批目标只报一次）
            if hit_count >= HIT_NEED and not alarm_latched and now - last_alarm > COOLDOWN:
                last_alarm = now
                alarm_latched = True
                alarm_count += 1
                best = max(targets, key=lambda d: d["conf"])
                x1, y1, x2, y2 = best["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tlat, tlon = px_to_gps(cx, cy)

                last_target = {"cls": best["cls"], "conf": best["conf"],
                               "px": (cx, cy), "lat": tlat, "lon": tlon}

                target_label.config(text=f"{best['cls']}   {best['conf']:.2f}\n"
                                         f"像素 ({cx}, {cy})\n"
                                         f"{tlat:.5f}, {tlon:.5f}")
                add_log(f"⚠ 发现异常！{best['cls']}   置信度 {best['conf']:.2f}")
                add_log(f"📍 目标位置：经度 {tlon:.5f}  纬度 {tlat:.5f}（像素 {cx},{cy}）")

                banner.config(text=f"⚠ 告警：{best['cls']}  {best['conf']:.2f}", bg="red")
                run_status.config(text="⚠ 告警中", fg="#ff5252")
                if SHOW_POPUP:
                    messagebox.showwarning("告警",
                                           f"检测到 {best['cls']}！\n置信度 {best['conf']:.2f}")
            elif hit_count > 0:
                banner.config(text=f"监测中：{targets[0]['cls']}  ({hit_count}/{HIT_NEED})",
                              bg="#d4a017")
            else:
                banner.config(text="状态正常", bg="#2e8b57")
                if not paused:
                    run_status.config(text="● 运行中", fg="#4caf50")

            # 巡航心跳
            if now - last_heartbeat > HEARTBEAT_SEC:
                last_heartbeat = now
                if not targets:
                    add_log("🛸 巡航中… 未发现异常")

            # 小地图
            tgt = (last_target["lat"], last_target["lon"]) if last_target else None
            draw_minimap(frame, tgt, car_gps, trail)

            show = cv2.resize(frame, (480, 360))
            rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            video_label.config(image=tk_img)
            video_label.image = tk_img

    # ========== ③ 统计信息（★ 不受暂停影响）==========
    fps_now = n / max(time.time() - t0, 1e-6)
    stat_label.config(text=f"已运行 {int(time.time() - t0)} 秒\n"
                           f"累计告警 {alarm_count} 次\n"
                           f"FPS {fps_now:.1f}")

    root.after(10, update)

t0 = time.time()
update()
root.mainloop()
