from ultralytics import YOLO

# Load the model using the best trained weights
model = YOLO("runs/segment/crack_model-5/weights/best.pt")

# Run prediction on test images and save the results
results = model.predict(source="dataset/test/images", show=True, save=True, conf=0.5)

print("Prediction and results visualization completed successfully!")