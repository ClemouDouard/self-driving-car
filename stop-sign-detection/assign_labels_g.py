import os
import cv2
import csv

STOP_DIR = "stop"
NOT_STOP_DIR = "not_stop"
OUTPUT_CSV = "annotations.csv"

drawing = False
ix, iy = -1, -1
box = []


def get_all_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, box, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        box = []

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box = [ix, iy, x, y]
        img_copy = img.copy()
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotate STOP Sign", img_copy)


def annotate_stop_images(image_paths):
    global img, img_copy, box
    annotations = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 480))
        img_copy = img.copy()
        box = []

        print(f"\n[INFO] Annotating: {img_path}")
        cv2.imshow("Annotate STOP Sign", img)
        cv2.setMouseCallback("Annotate STOP Sign", draw_rectangle)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # Enter key to save
                if box:
                    x_min = min(box[0], box[2])
                    y_min = min(box[1], box[3])
                    x_max = max(box[0], box[2])
                    y_max = max(box[1], box[3])
                    annotations.append([img_path, x_min, y_min, x_max, y_max, 1])
                    break
                else:
                    print("⚠️ No box selected. Please click two points.")
            elif key == ord('s'):  # skip image
                print("⏭ Skipping...")
                break
            elif key == 27:  # ESC to exit
                print("🚪 Exiting annotation.")
                cv2.destroyAllWindows()
                return annotations

    return annotations


def annotate_not_stop_images(image_paths):
    return [[img_path, -1, -1, -1, -1, 0] for img_path in image_paths]


def main():
    stop_images = get_all_images(STOP_DIR)
    not_stop_images = get_all_images(NOT_STOP_DIR)

    print(f"🛑 {len(stop_images)} STOP images to annotate.")
    stop_annots = annotate_stop_images(stop_images)

    print(f"🚫 {len(not_stop_images)} NON-STOP images.")
    not_stop_annots = annotate_not_stop_images(not_stop_images)

    all_annots = stop_annots + not_stop_annots

    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])
        writer.writerows(all_annots)

    print(f"\n✅ Saved annotations to {OUTPUT_CSV}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
