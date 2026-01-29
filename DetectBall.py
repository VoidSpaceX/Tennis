from ultralytics import YOLO

model = YOLO(r"C:\Users\User\Downloads\Практика\Модели\v3.pt")
results =  model.predict(
    source=r"C:\Users\User\Downloads\Практика\Практика\ТВД 50\ТВД 50.mp4",
    show=True,
    conf=0.15,
    imgsz=640
)
