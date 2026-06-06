import os
import cv2
from tqdm import tqdm
from label_parser import LabelParser

yolo_classes = {
    'bike': 0,
    'bus': 1,
    'car': 2,
    'motor': 3,
    'person': 4,
    'rider': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'train': 8,
    'truck': 9
}

def save_labels_in_yolo_format(label_json_path: str, image_dir_path: str, output_dir_path: str):
    os.makedirs(output_dir_path, exist_ok=True)

    parser = LabelParser(label_json_path, image_dir_path)
    for label in tqdm(parser.labels):
        image_name = label['name']
        image_path = f"{image_dir_path}/{image_name}"

        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        yolo_labels = []
        for label_info in label['labels']:
            label_category = label_info.get('category')
            if label_category in yolo_classes:
                class_id = yolo_classes[label_category]
                box2d = label_info.get('box2d', {})
                x1 = box2d.get('x1')
                y1 = box2d.get('y1')
                x2 = box2d.get('x2')
                y2 = box2d.get('y2')

                if None not in (x1, y1, x2, y2):
                    x_center = (x1 + x2) / 2 / image_width
                    y_center = (y1 + y2) / 2 / image_height
                    width = (x2 - x1) / image_width
                    height = (y2 - y1) / image_height
                    yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

        with open(f"{output_dir_path}/{image_name.replace('.jpg', '.txt')}", 'w') as f:
            for yolo_label in yolo_labels:
                f.write(f"{yolo_label}\n")

if __name__ == "__main__":
    label_json_path: str = "/media/mkd/Manav Desai/Datasets/bdd_dataset/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val_cleaned.json"
    image_dir_path: str = "/media/mkd/Manav Desai/Datasets/bdd_dataset/bdd100k_images_100k/bdd100k/images/100k/val"
    output_dir_path: str = "/media/mkd/Manav Desai/Datasets/bdd_dataset/bdd100k_images_100k/bdd100k/labels_yolo/100k/val"

    save_labels_in_yolo_format(label_json_path, image_dir_path, output_dir_path)
