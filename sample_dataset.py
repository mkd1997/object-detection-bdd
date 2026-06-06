import os
import math
import shutil
from tqdm import tqdm
from typing import Optional

import pandas as pd


def sample_unique_rows_by_category(
    df: pd.DataFrame,
    image_col: str = "image_name",
    category_col: str = "category",
    frac: float = 0.1,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    if image_col not in df.columns or category_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{image_col}' and '{category_col}' columns")

    if not 0 < frac <= 1:
        raise ValueError("frac must be between 0 and 1")

    categories = sorted(
        df[category_col].dropna().unique(),
        key=lambda c: df[df[category_col] == c][image_col].nunique(),
    )

    selected_images = []
    for index, category in tqdm(enumerate(categories)):
        cat_df = df[df[category_col] == category]
        target = max(1, int(math.ceil(len((cat_df[image_col].unique())) * frac)))
        cat_df: pd.DataFrame = cat_df.sample(
            frac=1,
            random_state=random_state + index if random_state is not None else None,
        )

        count = 0
        for _, row in cat_df.iterrows():
            image_name = row[image_col]
            if image_name not in selected_images:
                selected_images.append(image_name)
                count += 1
                if count >= target:
                    break

    return df.loc[df[image_col].isin(selected_images)].reset_index(drop=True)


def sample_csv(
    input_csv_path: str,
    output_csv_path: str,
    frac: float = 0.1,
    image_col: str = "image_name",
    category_col: str = "category",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    sampled = sample_unique_rows_by_category(
        df,
        image_col=image_col,
        category_col=category_col,
        frac=frac,
        random_state=random_state,
    )
    sampled.to_csv(output_csv_path, index=False)
    return sampled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample 10% rows per category with unique images across categories."
    )
    parser.add_argument("--input_csv", help="path to the source CSV file")
    parser.add_argument("--output_csv", help="path to the sampled output CSV file")
    parser.add_argument("--frac", type=float, default=0.1, help="fraction of rows to sample for each category")
    parser.add_argument("--image_col", default="image_name", help="image name column")
    parser.add_argument("--category_col", default="label_category", help="category column")
    parser.add_argument("--random_state", type=int, default=42, help="random seed")
    parser.add_argument("--data_dir", required=True, help="base directory for images and labels")
    parser.add_argument("--split", required=True, help="dataset split to sample (e.g. train, val, test)")
    args = parser.parse_args()

    sample_df = sample_csv(
        args.input_csv,
        args.output_csv,
        frac=args.frac,
        image_col=args.image_col,
        category_col=args.category_col,
        random_state=args.random_state,
    )

    output_image_dir = f"{os.path.dirname(args.output_csv)}/images/{args.split}"
    output_label_dir = f"{os.path.dirname(args.output_csv)}/labels/{args.split}"
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for image_name in tqdm(sample_df[args.image_col].unique()):
        image_path = os.path.join(args.data_dir, "images/100k", args.split, image_name)
        label_path = os.path.join(args.data_dir, "labels/100k", args.split, image_name.replace(".jpg", ".txt"))
        if os.path.exists(image_path) and os.path.exists(label_path):
            shutil.copy(image_path, os.path.join(output_image_dir, image_name))
            shutil.copy(label_path, os.path.join(output_label_dir, image_name.replace(".jpg", ".txt")))
        else:
            print(f"Warning: Missing image or label for {image_name}")