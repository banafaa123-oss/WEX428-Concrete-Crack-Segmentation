from ultralytics import YOLO

def main():
    # Load YOLOv8 segmentation model for fine-tuning
    model = YOLO("yolov8n-seg.pt")

    # Start training on the custom crack dataset
    results = model.train(
        data="dataset/data.yaml",  # Path to the dataset configuration file
        epochs=50,
        imgsz=640,
        batch=16,
        name="crack_model"
    )

    print("Training completed successfully!")

if __name__ == "__main__":
    main()