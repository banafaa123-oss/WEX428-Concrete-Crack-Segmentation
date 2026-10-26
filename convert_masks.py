import os
import cv2
from glob import glob

def convert_masks_to_yolo(split):
    img_dir = f"dataset/{split}/images"
    mask_dir = f"dataset/{split}/masks"
    label_dir = f"dataset/{split}/labels"
    os.makedirs(label_dir, exist_ok=True)
    
    mask_paths = glob(os.path.join(mask_dir, "*.*"))
    for mask_path in mask_paths:
        filename = os.path.splitext(os.path.basename(mask_path))[0]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        
        # استخراج الحدود من القناع الثنائي
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        txt_path = os.path.join(label_dir, f"{filename}.txt")
        with open(txt_path, "w") as f:
            for cnt in contours:
                if cv2.contourArea(cnt) < 5:  # تجاهل الأجزاء الصغيرة جداً
                    continue
                polygon = []
                for point in cnt:
                    px, py = point[0]
                    polygon.append(f"{px / w:.6f}")
                    polygon.append(f"{py / h:.6f}")
                if len(polygon) >= 6:
                    line = f"0 " + " ".join(polygon) + "\n"
                    f.write(line)

if __name__ == "__main__":
    for split in ['train', 'test']:
        if os.path.exists(f"dataset/{split}/masks"):
            print(f"Converting {split} masks...")
            convert_masks_to_yolo(split)
    print("Conversion completed successfully!")