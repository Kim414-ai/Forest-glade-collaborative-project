# 森林空地协同巡检系统 · 界面与联动模块

基于 YOLOv8n 的森林火情巡检系统：摄像头 / 航拍视频 → 烟雾识别 → 经纬度定位
→ 任务下发 → 无人车联动 → 全程日志留档。

## 功能
- 实时烟雾检测（自训练 YOLOv8n，单类别 smoke）
- 双视频源切换（摄像头 / 航拍视频）
- 连续帧告警 + 事件锁存（同一批目标只报一次）
- 像素坐标 → 模拟 GPS 换算
- 区域小地图 HUD（目标点 / 无人车位置 / 轨迹）
- 任务下发（JSON 文件接口）
- 日志留档（alarm_log.txt）

## 快速开始
```bash
python -m pip install -r requirements.txt
python ugv_sim.py     # 启动无人车模拟器
python tk_ui.py       # 启动巡检界面
