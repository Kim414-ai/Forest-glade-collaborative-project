import time
import json
import os

TASK_FILE   = "ugv_tasks.txt"
STATUS_FILE = "ugv_status.txt"

STEP = 8              # 每步前进 8%（约 13 步到达）
TICK = 0.4            # 每 0.4 秒走一步

# ★ 小车初始位置：改成"区域左下角内部"（区域范围内）
START_LAT = 30.49940
START_LON = 114.30010

state     = "idle"
progress  = 0
target    = None
start_pos = {"lat": START_LAT, "lon": START_LON}
cur_pos   = {"lat": START_LAT, "lon": START_LON}
task_cls  = ""

def count_tasks():
    if not os.path.exists(TASK_FILE):
        return 0
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        return len([l for l in f if l.strip()])

processed = count_tasks()      # 启动时忽略历史任务

def write_status():
    data = {
        "time":     time.strftime("%H:%M:%S"),
        "state":    state,
        "progress": progress,
        "cls":      task_cls,
        "current":  {"lat": round(cur_pos["lat"], 6), "lon": round(cur_pos["lon"], 6)},
        "target":   target,
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def read_new_task():
    global processed
    if not os.path.exists(TASK_FILE):
        return None
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) <= processed:
        return None
    processed = len(lines)
    return json.loads(lines[-1])

print(f"小车模拟器已启动（忽略 {processed} 条历史任务），等待新任务...")
write_status()

while True:
    if state in ("idle", "arrived"):
        task = read_new_task()
        if task is not None:
            target   = task["geo"]
            task_cls = task.get("cls", "")
            start_pos = dict(cur_pos)
            progress = 0
            state = "going"
            print(f"[{time.strftime('%H:%M:%S')}] 收到任务 → "
                  f"目标 ({target['lat']}, {target['lon']})")
            write_status()

    elif state == "going":
        progress += STEP
        if progress >= 100:
            progress = 100
            cur_pos = dict(target)
            state = "arrived"
            print(f"[{time.strftime('%H:%M:%S')}] ✅ 已到达目标点")
        else:
            k = progress / 100.0
            cur_pos["lat"] = start_pos["lat"] + (target["lat"] - start_pos["lat"]) * k
            cur_pos["lon"] = start_pos["lon"] + (target["lon"] - start_pos["lon"]) * k
            print(f"[{time.strftime('%H:%M:%S')}] 前往中 {progress}%  "
                  f"({cur_pos['lat']:.5f}, {cur_pos['lon']:.5f})")
        write_status()

    time.sleep(TICK)
