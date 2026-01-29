from ultralytics import YOLO

model = YOLO(r"")
results =  model.track(
    source=r"",
    show=True,
    tracker='bytetrack.yaml',
    persist=True,
    conf=0.1,
    imgsz=360
)
