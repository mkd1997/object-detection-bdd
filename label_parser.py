import cv2
import json
import pandas as pd
from tqdm import tqdm

EXCLUDED_LABEL_CATEGORIES: tuple[str, ...] = ("drivable area", "lane")


class LabelParser:
    def __init__(self, label_json_file_path, data_dir_path, keep_raw_label_list: bool = True):
        self.label_json_file_path: str = label_json_file_path
        self.data_dir_path: str = data_dir_path

        with open(label_json_file_path, 'r') as f:
            raw_label_list: list = json.load(f)
        self.raw_label_list: list = raw_label_list if keep_raw_label_list else []

        cleaned_label_list: list = []
        for label in tqdm(raw_label_list):
            cleaned_label_dict: dict = {}
            for key, value in label.items():
                cleaned_key: str = key.strip()

                if ((cleaned_key == 'labels') and isinstance(value, list)):
                    cleaned_labels: list = []
                    for label in value:
                        if (label['category'] not in EXCLUDED_LABEL_CATEGORIES):
                            cleaned_labels.append(label)

                    cleaned_value: list = cleaned_labels
                else:
                    cleaned_value = value.strip() if isinstance(value, str) else value
                cleaned_label_dict[cleaned_key] = cleaned_value

            cleaned_label_list.append(cleaned_label_dict)

        self.labels: list = cleaned_label_list

    def write_to_json(self, output_file_path):
        with open(output_file_path, 'w') as f:
            json.dump(self.labels, f, indent=4)

    def get_object_df(self, split: str | None = None):
        object_rows: list[dict] = []

        for label in tqdm(self.labels):
            image_name: str = label['name']
            weather_condition: str = label['attributes']['weather']
            scene_category: str = label['attributes']['scene']
            time_of_day: str = label['attributes']['timeofday']

            for label_info in label['labels']:
                box2d: dict = label_info.get('box2d', {})
                x1 = box2d.get('x1')
                y1 = box2d.get('y1')
                x2 = box2d.get('x2')
                y2 = box2d.get('y2')
                box_width = x2 - x1 if x1 is not None and x2 is not None else None
                box_height = y2 - y1 if y1 is not None and y2 is not None else None
                box_area = (
                    box_width * box_height
                    if box_width is not None and box_height is not None
                    else None
                )
                label_attributes: dict = label_info.get('attributes', {})

                row = {
                    'image_name': image_name,
                    'weather_condition': weather_condition,
                    'scene_category': scene_category,
                    'time_of_day': time_of_day,
                    'label_id': label_info.get('id'),
                    'label_category': label_info.get('category'),
                    'label_occlusion': label_attributes.get('occluded'),
                    'label_truncation': label_attributes.get('truncated'),
                    'label_traffic_light_color': label_attributes.get('trafficLightColor'),
                    'label_x1': x1,
                    'label_y1': y1,
                    'label_x2': x2,
                    'label_y2': y2,
                    'bbox_width': box_width,
                    'bbox_height': box_height,
                    'bbox_area': box_area,
                }
                if split is not None:
                    row['split'] = split
                object_rows.append(row)

        return pd.DataFrame(object_rows)

    def get_image_df(self, split: str | None = None):
        image_rows: list[dict] = []

        for label in tqdm(self.labels):
            row = {
                'image_name': label['name'],
                'weather_condition': label['attributes']['weather'],
                'scene_category': label['attributes']['scene'],
                'time_of_day': label['attributes']['timeofday'],
                'retained_object_count': len(label['labels']),
            }
            if split is not None:
                row['split'] = split
            image_rows.append(row)

        return pd.DataFrame(image_rows)

if __name__ == "__main__":
    label_parser = LabelParser("/media/mkd/Manav Desai/Datasets/bdd_dataset/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json", "/media/mkd/Manav Desai/Datasets/bdd_dataset/bdd100k_images_100k/bdd100k/images/100k/train")
    label_parser.write_to_json("/media/mkd/Manav Desai/Datasets/bdd_dataset/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train_cleaned.json")
    label_df = label_parser.get_label_df()
    label_df.to_csv("/media/mkd/Manav Desai/Datasets/bdd_dataset/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train_cleaned.csv", index=False)
