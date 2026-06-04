import argparse
import html
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bdd-report")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from label_parser import EXCLUDED_LABEL_CATEGORIES, LabelParser


DATASET_ROOT = Path("/media/mkd/Manav Desai/Datasets/bdd_dataset")
LABEL_DIR = DATASET_ROOT / "bdd100k_labels_release/bdd100k/labels"
IMAGE_DIR = DATASET_ROOT / "bdd100k_images_100k/bdd100k/images/100k"

DEFAULT_TRAIN_LABELS = LABEL_DIR / "bdd100k_labels_images_train.json"
DEFAULT_VAL_LABELS = LABEL_DIR / "bdd100k_labels_images_val.json"
DEFAULT_TRAIN_IMAGES = IMAGE_DIR / "train"
DEFAULT_VAL_IMAGES = IMAGE_DIR / "val"
DEFAULT_OUTPUT_DIR = Path("reports/bdd_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static BDD100K train/val label analysis webpage."
    )
    parser.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--val-labels", type=Path, default=DEFAULT_VAL_LABELS)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--val-images", type=Path, default=DEFAULT_VAL_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Allow running outside the bdd conda environment.",
    )
    return parser.parse_args()


def require_bdd_environment(skip_env_check: bool) -> None:
    if skip_env_check:
        return

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env != "bdd":
        raise RuntimeError(
            "Run this script in the bdd conda environment, for example: "
            "conda run -n bdd python generate_bdd_report.py"
        )


def load_split(label_path: Path, image_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading {split} labels from {label_path}")
    parser = LabelParser(
        str(label_path),
        str(image_dir),
        keep_raw_label_list=False,
    )
    object_df = parser.get_object_df(split=split)
    image_df = parser.get_image_df(split=split)
    return object_df, image_df


def save_figure(fig: plt.Figure, assets_dir: Path, filename: str) -> str:
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"assets/{filename}"


def format_int(value: float | int) -> str:
    return f"{int(value):,}"


def figure_size(width: float = 12, height: float = 6) -> tuple[float, float]:
    return (width, height)


def plot_grouped_bar(
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    assets_dir: Path,
    filename: str,
    rotate: int = 35,
) -> str:
    fig, ax = plt.subplots(figsize=figure_size(max(10, len(data.index) * 0.8), 6))
    data.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="split")
    fig.tight_layout()
    return save_figure(fig, assets_dir, filename)


def plot_stacked_positive_negative(
    positive_negative_df: pd.DataFrame,
    split: str,
    assets_dir: Path,
) -> str:
    split_df = positive_negative_df[positive_negative_df["split"] == split].copy()
    split_df = split_df.sort_values("positive_images", ascending=False)
    plot_df = split_df.set_index("label_category")[["positive_images", "negative_images"]]

    fig, ax = plt.subplots(figsize=figure_size(max(10, len(plot_df.index) * 0.8), 6))
    plot_df.plot(kind="bar", stacked=True, ax=ax, color=["#2f7ed8", "#d9d9d9"])
    ax.set_title(f"{split.title()} per-class positive vs negative images")
    ax.set_xlabel("")
    ax.set_ylabel("image count")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["positive", "negative"])
    fig.tight_layout()
    return save_figure(fig, assets_dir, f"{split}_positive_negative_images.png")


def plot_attribute_heatmap(
    object_df: pd.DataFrame,
    split: str,
    attribute: str,
    assets_dir: Path,
) -> str | None:
    split_df = object_df[object_df["split"] == split]
    if split_df.empty or attribute not in split_df:
        return None

    counts = pd.crosstab(split_df["label_category"], split_df[attribute].astype(str))
    if counts.empty:
        return None

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages = percentages.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    fig_width = max(9, len(percentages.columns) * 1.3)
    fig_height = max(5, len(percentages.index) * 0.45)
    fig, ax = plt.subplots(figsize=figure_size(fig_width, fig_height))
    image = ax.imshow(percentages.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_title(f"{split.title()} {attribute} distribution by label category")
    ax.set_xticks(range(len(percentages.columns)))
    ax.set_xticklabels(percentages.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(percentages.index)))
    ax.set_yticklabels(percentages.index)

    for row_idx in range(percentages.shape[0]):
        for col_idx in range(percentages.shape[1]):
            value = percentages.iat[row_idx, col_idx]
            if value >= 5:
                color = "white" if value >= 55 else "black"
                ax.text(col_idx, row_idx, f"{value:.0f}%", ha="center", va="center", color=color, fontsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("row percentage")
    fig.tight_layout()
    safe_attribute = attribute.replace("_", "-")
    return save_figure(fig, assets_dir, f"{split}_{safe_attribute}_by_category.png")


def plot_objects_per_image(image_df: pd.DataFrame, assets_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=figure_size(10, 6))
    max_count = int(image_df["retained_object_count"].max())
    bins = range(0, min(max_count, 80) + 2)
    for split, split_df in image_df.groupby("split"):
        ax.hist(
            split_df["retained_object_count"].clip(upper=80),
            bins=bins,
            alpha=0.55,
            label=split,
        )
    ax.set_title("Objects per image distribution")
    ax.set_xlabel("retained object count per image, clipped at 80")
    ax.set_ylabel("image count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="split")
    fig.tight_layout()
    return save_figure(fig, assets_dir, "objects_per_image.png")


def plot_bbox_area_by_class(object_df: pd.DataFrame, assets_dir: Path) -> str | None:
    bbox_df = object_df.dropna(subset=["bbox_area"])
    if bbox_df.empty:
        return None

    top_categories = bbox_df["label_category"].value_counts().head(12).index.tolist()
    plot_df = bbox_df[bbox_df["label_category"].isin(top_categories)].copy()
    plot_df["bbox_area"] = plot_df["bbox_area"].clip(lower=0)

    grouped = [plot_df.loc[plot_df["label_category"] == category, "bbox_area"] for category in top_categories]
    fig, ax = plt.subplots(figsize=figure_size(12, 6))
    ax.boxplot(grouped, tick_labels=top_categories, showfliers=False)
    ax.set_title("Bounding-box area distribution by class")
    ax.set_xlabel("")
    ax.set_ylabel("bbox area in pixels, outliers hidden")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, assets_dir, "bbox_area_by_class.png")


def plot_average_bbox_area(avg_bbox_df: pd.DataFrame, assets_dir: Path) -> str:
    pivot = avg_bbox_df.pivot(index="label_category", columns="split", values="avg_bbox_area")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    return plot_grouped_bar(
        pivot,
        "Average bounding-box area by class",
        "average bbox area in pixels",
        assets_dir,
        "average_bbox_area_by_class.png",
    )


def plot_center_heatmap(object_df: pd.DataFrame, split: str, assets_dir: Path) -> str | None:
    split_df = object_df[object_df["split"] == split].dropna(
        subset=["label_x1", "label_y1", "label_x2", "label_y2"]
    )
    if split_df.empty:
        return None

    center_x = (split_df["label_x1"] + split_df["label_x2"]) / 2
    center_y = (split_df["label_y1"] + split_df["label_y2"]) / 2

    fig, ax = plt.subplots(figsize=figure_size(9, 5))
    heatmap = ax.hist2d(center_x, center_y, bins=[24, 14], cmap="magma")
    ax.invert_yaxis()
    ax.set_title(f"{split.title()} object center density")
    ax.set_xlabel("x center")
    ax.set_ylabel("y center")
    colorbar = fig.colorbar(heatmap[3], ax=ax)
    colorbar.set_label("object count")
    fig.tight_layout()
    return save_figure(fig, assets_dir, f"{split}_object_center_density.png")


def plot_class_center_heatmaps(
    object_df: pd.DataFrame,
    split: str,
    assets_dir: Path,
) -> str | None:
    split_df = object_df[object_df["split"] == split].dropna(
        subset=["label_x1", "label_y1", "label_x2", "label_y2", "label_category"]
    )
    if split_df.empty:
        return None

    categories = split_df["label_category"].value_counts().index.tolist()
    column_count = 3
    row_count = math.ceil(len(categories) / column_count)
    x_max = split_df["label_x2"].max()
    y_max = split_df["label_y2"].max()

    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=figure_size(column_count * 4.6, row_count * 3.2),
        squeeze=False,
    )
    axes_list = axes.ravel()

    for ax, category in zip(axes_list, categories):
        category_df = split_df[split_df["label_category"] == category]
        center_x = (category_df["label_x1"] + category_df["label_x2"]) / 2
        center_y = (category_df["label_y1"] + category_df["label_y2"]) / 2
        ax.hist2d(
            center_x,
            center_y,
            bins=[24, 14],
            range=[[0, x_max], [0, y_max]],
            cmap="magma",
        )
        ax.invert_yaxis()
        ax.set_title(f"{category} ({len(category_df):,})")
        ax.set_xlabel("x center")
        ax.set_ylabel("y center")

    for ax in axes_list[len(categories):]:
        ax.axis("off")

    fig.suptitle(f"{split.title()} object center density by class", fontsize=16)
    fig.tight_layout()
    return save_figure(fig, assets_dir, f"{split}_class_object_center_density.png")


def build_category_summary(object_df: pd.DataFrame, image_df: pd.DataFrame) -> pd.DataFrame:
    object_counts = (
        object_df.groupby(["split", "label_category"])
        .size()
        .rename("object_count")
        .reset_index()
    )
    image_counts = (
        object_df.drop_duplicates(["split", "label_category", "image_name"])
        .groupby(["split", "label_category"])
        .size()
        .rename("positive_images")
        .reset_index()
    )
    summary = object_counts.merge(image_counts, on=["split", "label_category"], how="outer").fillna(0)

    total_objects = object_df.groupby("split").size().rename("total_objects")
    total_images = image_df.groupby("split").size().rename("total_images")
    summary = summary.merge(total_objects, on="split", how="left")
    summary = summary.merge(total_images, on="split", how="left")
    summary["object_pct"] = summary["object_count"] / summary["total_objects"] * 100
    summary["positive_image_pct"] = summary["positive_images"] / summary["total_images"] * 100
    return summary.sort_values(["split", "object_count"], ascending=[True, False])


def build_positive_negative_summary(
    object_df: pd.DataFrame,
    image_df: pd.DataFrame,
) -> pd.DataFrame:
    categories = sorted(object_df["label_category"].dropna().unique())
    total_images_by_split = image_df.groupby("split").size().to_dict()
    positive = (
        object_df.drop_duplicates(["split", "label_category", "image_name"])
        .groupby(["split", "label_category"])
        .size()
        .rename("positive_images")
        .reset_index()
    )
    positive_lookup = {
        (row["split"], row["label_category"]): int(row["positive_images"])
        for _, row in positive.iterrows()
    }

    rows = []
    for split, total_images in total_images_by_split.items():
        for category in categories:
            positive_images = positive_lookup.get((split, category), 0)
            rows.append(
                {
                    "split": split,
                    "label_category": category,
                    "positive_images": positive_images,
                    "negative_images": total_images - positive_images,
                    "total_images": total_images,
                    "positive_image_pct": positive_images / total_images * 100,
                }
            )
    return pd.DataFrame(rows)


def build_drift_table(category_summary: pd.DataFrame) -> pd.DataFrame:
    object_pct = category_summary.pivot(
        index="label_category",
        columns="split",
        values="object_pct",
    ).fillna(0)
    image_pct = category_summary.pivot(
        index="label_category",
        columns="split",
        values="positive_image_pct",
    ).fillna(0)

    for column in ["train", "val"]:
        if column not in object_pct:
            object_pct[column] = 0
        if column not in image_pct:
            image_pct[column] = 0

    drift = pd.DataFrame(
        {
            "label_category": object_pct.index,
            "train_object_pct": object_pct["train"].values,
            "val_object_pct": object_pct["val"].values,
            "object_pct_point_diff_val_minus_train": (
                object_pct["val"] - object_pct["train"]
            ).values,
            "train_positive_image_pct": image_pct["train"].values,
            "val_positive_image_pct": image_pct["val"].values,
            "image_pct_point_diff_val_minus_train": (
                image_pct["val"] - image_pct["train"]
            ).values,
        }
    )
    return drift.sort_values(
        "object_pct_point_diff_val_minus_train",
        key=lambda series: series.abs(),
        ascending=False,
    )


def plot_category_drift(drift_df: pd.DataFrame, assets_dir: Path) -> str:
    plot_df = drift_df.set_index("label_category")[
        "object_pct_point_diff_val_minus_train"
    ].sort_values()
    fig, ax = plt.subplots(figsize=figure_size(10, max(5, len(plot_df) * 0.45)))
    colors = ["#c43c39" if value < 0 else "#2f7ed8" for value in plot_df]
    plot_df.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Train vs val class drift")
    ax.set_xlabel("object percentage-point difference, val minus train")
    ax.set_ylabel("")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, assets_dir, "train_val_class_drift.png")


def build_attribute_imbalance_table(object_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, split_df in object_df.groupby("split"):
        for attribute in ["weather_condition", "scene_category", "time_of_day"]:
            counts = pd.crosstab(split_df["label_category"], split_df[attribute].astype(str))
            for category, row in counts.iterrows():
                total = int(row.sum())
                if total == 0:
                    continue
                top_value = row.idxmax()
                top_count = int(row.max())
                rows.append(
                    {
                        "split": split,
                        "label_category": category,
                        "attribute": attribute,
                        "dominant_value": top_value,
                        "dominant_count": top_count,
                        "dominant_pct": top_count / total * 100,
                        "object_count": total,
                    }
                )
    return pd.DataFrame(rows).sort_values("dominant_pct", ascending=False)


def dataframe_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)

    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.2f}")
        elif pd.api.types.is_integer_dtype(display_df[column]):
            display_df[column] = display_df[column].map(format_int)

    return display_df.to_html(index=False, classes="data-table", border=0)


def render_html(
    output_path: Path,
    chart_paths: dict[str, str | None],
    tables: dict[str, pd.DataFrame],
    train_labels: Path,
    val_labels: Path,
) -> None:
    def image_block(title: str, chart_key: str) -> str:
        chart_path = chart_paths.get(chart_key)
        if not chart_path:
            return ""
        return (
            f"<section><h2>{html.escape(title)}</h2>"
            f'<img src="{html.escape(chart_path)}" alt="{html.escape(title)}"></section>'
        )

    attribute_sections = []
    for split in ["train", "val"]:
        for attribute in [
            "weather_condition",
            "scene_category",
            "time_of_day",
            "label_occlusion",
            "label_truncation",
            "label_traffic_light_color",
        ]:
            title = f"{split.title()} {attribute.replace('_', ' ')} by class"
            attribute_sections.append(image_block(title, f"{split}_{attribute}"))

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BDD100K Train/Val Label Analysis</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #202124;
      --muted: #5f6368;
      --border: #dadce0;
      --surface: #ffffff;
      --band: #f6f8fa;
      --accent: #1967d2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
      color: var(--text);
      background: var(--surface);
    }}
    header {{
      padding: 32px 40px 24px;
      border-bottom: 1px solid var(--border);
      background: var(--band);
    }}
    main {{ padding: 24px 40px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; }}
    h3 {{ margin: 24px 0 12px; font-size: 18px; }}
    p {{ max-width: 1100px; }}
    section {{
      padding: 24px 0;
      border-bottom: 1px solid var(--border);
    }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      background: white;
    }}
    .meta {{
      color: var(--muted);
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      max-width: 960px;
    }}
    .stat {{
      border: 1px solid var(--border);
      padding: 14px 16px;
      background: white;
    }}
    .stat strong {{
      display: block;
      font-size: 24px;
      margin-bottom: 4px;
    }}
    .table-wrap {{
      overflow-x: auto;
      max-width: 100%;
    }}
    table.data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
      background: white;
    }}
    .data-table th,
    .data-table td {{
      border: 1px solid var(--border);
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    .data-table th:first-child,
    .data-table td:first-child,
    .data-table th:nth-child(2),
    .data-table td:nth-child(2) {{
      text-align: left;
    }}
    .note {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>BDD100K Train/Val Label Analysis</h1>
    <p class="meta">Train labels: {html.escape(str(train_labels))}</p>
    <p class="meta">Val labels: {html.escape(str(val_labels))}</p>
    <p class="meta">Excluded object categories: {html.escape(", ".join(EXCLUDED_LABEL_CATEGORIES))}</p>
  </header>
  <main>
    <section>
      <h2>Dataset Summary</h2>
      <div class="grid">
        <div class="stat"><strong>{format_int(tables["split_summary"].loc["train", "images"])}</strong>train images</div>
        <div class="stat"><strong>{format_int(tables["split_summary"].loc["val", "images"])}</strong>val images</div>
        <div class="stat"><strong>{format_int(tables["split_summary"].loc["train", "objects"])}</strong>train retained objects</div>
        <div class="stat"><strong>{format_int(tables["split_summary"].loc["val", "objects"])}</strong>val retained objects</div>
      </div>
      <h3>Split Summary</h3>
      <div class="table-wrap">{dataframe_to_html(tables["split_summary"].reset_index())}</div>
    </section>

    {image_block("Label category object-count distribution", "category_counts")}
    {image_block("Images per class", "images_per_class")}
    {image_block("Objects per image", "objects_per_image")}
    {image_block("Average bounding-box area by class", "avg_bbox_area")}
    {image_block("Bounding-box area distribution", "bbox_area")}
    {image_block("Train object center density", "train_center_heatmap")}
    {image_block("Train object center density by class", "train_class_center_heatmaps")}
    {image_block("Val object center density", "val_center_heatmap")}
    {image_block("Train per-class positive vs negative images", "train_positive_negative")}
    {image_block("Val per-class positive vs negative images", "val_positive_negative")}

    <section>
      <h2>Attribute Distributions By Class</h2>
      <p class="note">Each heatmap row is normalized within one label category.</p>
    </section>
    {"".join(attribute_sections)}

    <section>
      <h2>Category Comparison Table</h2>
      <div class="table-wrap">{dataframe_to_html(tables["category_summary"])}</div>
    </section>
    <section>
      <h2>Per-Class Positive/Negative Image Counts</h2>
      <div class="table-wrap">{dataframe_to_html(tables["positive_negative"])}</div>
    </section>
    <section>
      <h2>Largest Train/Val Class Drift</h2>
      <div class="table-wrap">{dataframe_to_html(tables["drift"], max_rows=20)}</div>
    </section>
    <section>
      <h2>Attribute Imbalance Candidates</h2>
      <p class="note">Rows with high dominant percentages indicate class coverage concentrated in one condition.</p>
      <div class="table-wrap">{dataframe_to_html(tables["attribute_imbalance"], max_rows=30)}</div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_bdd_environment(args.skip_env_check)

    output_dir = args.output_dir
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    train_objects, train_images = load_split(args.train_labels, args.train_images, "train")
    val_objects, val_images = load_split(args.val_labels, args.val_images, "val")

    object_df = pd.concat([train_objects, val_objects], ignore_index=True)
    image_df = pd.concat([train_images, val_images], ignore_index=True)

    if object_df["label_category"].isin(EXCLUDED_LABEL_CATEGORIES).any():
        raise AssertionError("Excluded categories are present in object-level data.")

    split_summary = pd.DataFrame(
        {
            "images": image_df.groupby("split").size(),
            "objects": object_df.groupby("split").size(),
            "positive_images_any_retained_object": image_df.groupby("split")[
                "retained_object_count"
            ].apply(lambda series: int((series > 0).sum())),
        }
    )
    split_summary["negative_images_no_retained_objects"] = (
        split_summary["images"] - split_summary["positive_images_any_retained_object"]
    )
    split_summary["objects_per_image_mean"] = (
        object_df.groupby("split").size() / image_df.groupby("split").size()
    )

    category_summary = build_category_summary(object_df, image_df)
    positive_negative = build_positive_negative_summary(object_df, image_df)
    drift = build_drift_table(category_summary)
    attribute_imbalance = build_attribute_imbalance_table(object_df)

    positive_negative_check = positive_negative[
        positive_negative["positive_images"] + positive_negative["negative_images"]
        != positive_negative["total_images"]
    ]
    if not positive_negative_check.empty:
        raise AssertionError("Positive + negative image counts do not equal total images.")

    category_counts = (
        category_summary.pivot(index="label_category", columns="split", values="object_count")
        .fillna(0)
        .astype(int)
    )
    category_counts = category_counts.loc[category_counts.sum(axis=1).sort_values(ascending=False).index]

    images_per_class = (
        category_summary.pivot(index="label_category", columns="split", values="positive_images")
        .fillna(0)
        .astype(int)
    )
    images_per_class = images_per_class.loc[
        images_per_class.sum(axis=1).sort_values(ascending=False).index
    ]

    avg_bbox_area = (
        object_df.dropna(subset=["bbox_area"])
        .groupby(["split", "label_category"])["bbox_area"]
        .mean()
        .rename("avg_bbox_area")
        .reset_index()
    )

    chart_paths: dict[str, str | None] = {
        "category_counts": plot_grouped_bar(
            category_counts,
            "Label category object-count distribution",
            "object count",
            assets_dir,
            "label_category_object_counts.png",
        ),
        "images_per_class": plot_grouped_bar(
            images_per_class,
            "Images containing each class",
            "positive image count",
            assets_dir,
            "images_per_class.png",
        ),
        "objects_per_image": plot_objects_per_image(image_df, assets_dir),
        "bbox_area": plot_bbox_area_by_class(object_df, assets_dir),
        "category_drift": plot_category_drift(drift, assets_dir),
        "train_center_heatmap": plot_center_heatmap(object_df, "train", assets_dir),
        "train_class_center_heatmaps": plot_class_center_heatmaps(
            object_df,
            "train",
            assets_dir,
        ),
        "val_center_heatmap": plot_center_heatmap(object_df, "val", assets_dir),
        "train_positive_negative": plot_stacked_positive_negative(
            positive_negative,
            "train",
            assets_dir,
        ),
        "val_positive_negative": plot_stacked_positive_negative(
            positive_negative,
            "val",
            assets_dir,
        ),
    }

    if not avg_bbox_area.empty:
        chart_paths["avg_bbox_area"] = plot_average_bbox_area(avg_bbox_area, assets_dir)
    else:
        chart_paths["avg_bbox_area"] = None

    for split in ["train", "val"]:
        for attribute in [
            "weather_condition",
            "scene_category",
            "time_of_day",
            "label_occlusion",
            "label_truncation",
            "label_traffic_light_color",
        ]:
            chart_paths[f"{split}_{attribute}"] = plot_attribute_heatmap(
                object_df,
                split,
                attribute,
                assets_dir,
            )

    tables = {
        "split_summary": split_summary,
        "category_summary": category_summary,
        "positive_negative": positive_negative.sort_values(
            ["split", "positive_images"],
            ascending=[True, False],
        ),
        "drift": drift,
        "attribute_imbalance": attribute_imbalance,
    }

    index_path = output_dir / "index.html"
    render_html(index_path, chart_paths, tables, args.train_labels, args.val_labels)
    print(f"Wrote report to {index_path}")


if __name__ == "__main__":
    main()
