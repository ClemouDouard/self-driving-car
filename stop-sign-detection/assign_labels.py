import os
import cv2
import csv

# === CONFIG ===
STOP_DIR = "stop"
NOT_STOP_DIR = "not_stop"
OUTPUT_CSV = "annotations.csv"


# === UTILS ===
def get_all_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]


def annotate_stop_images(image_paths):
    annotations = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        resized = cv2.resize(img, (640, 480))
        cv2.imshow("Image", resized)
        print(f"\nAnnotating: {img_path}")
        print("Enter bounding box for STOP sign as: x_min y_min x_max y_max")
        print("Or enter 's' to skip this image.")

        key = input(">> ")
        if key.lower() == 's':
            continue

        try:
            x_min, y_min, x_max, y_max = map(int, key.strip().split())
            annotations.append([img_path, x_min, y_min, x_max, y_max, 1])
        except:
            print("Invalid input! Skipping this image.")
            continue
        cv2.destroyAllWindows()
    return annotations


def annotate_not_stop_images(image_paths):
    annotations = []
    for img_path in image_paths:
        annotations.append([img_path, -1, -1, -1, -1, 0])  # No box, just label
    return annotations


# === MAIN ===
def main():
    stop_images = get_all_images(STOP_DIR)
    not_stop_images = get_all_images(NOT_STOP_DIR)

    print(f"Found {len(stop_images)} STOP images to annotate.")
    stop_annots = annotate_stop_images(stop_images)

    print(f"Found {len(not_stop_images)} NON-STOP images.")
    not_stop_annots = annotate_not_stop_images(not_stop_images)

    all_annots = stop_annots + not_stop_annots

    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])
        writer.writerows(all_annots)

    print(f"\n✅ Saved annotations to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
