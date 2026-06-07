# Object Detection on BDD Dataset
This repository is for
1. Analysing the BDD Dataset. 
2. Training a YOLO26 model on a subset of the BDD Dataset
3. Evaluating the performance of the model

## Repository Setup
```
git clone https://github.com/mkd1997/object-detection-bdd.git
cd object-detection-bdd
conda env create -f bdd_env.yml
conda activate bdd
```

## Data Analysis
[generate_bdd_report.py](generate_bdd_report.py) has been used to generate data analysis report for the BDD Dataset.
[label_parser.py](label_parser.py) defines the helper class for generating cleaned labels after removing the `drivable area` and `lane` classes.

### Setup
To view the generated report, either open [index.html](reports/bdd_analysis/index.html) in your browser 

**OR**

```
docker run -d --name report-viewer -p 8080:80 -v ./reports/bdd_analysis:/usr/share/nginx/html:ro nginx:alpine
```

**NOTE:** In case of any change to the script or dataset, run the [generate_bdd_report.py](generate_bdd_report.py) and then run the docker image.

### Observation and Insights
Detailed observations and insights on the BDD Dataset can be found in [DataAnalysis.md](DataAnalysis.md)

## Model Development
Trained a YOLO26-Nano model on a subset of the full dataset.

### Data Preparatation
The following steps were followed for preparing the dataset for model training.
1. Converting the JSON labels to YOLO format using the [convert_bdd_to_yolo.py](convert_bdd_to_yolo.py) script. This involves getting the JSON annotations for each image and then writing them to a text file after normalizing them. The YOLO format is `<class-id> <x> <y> <width> <height>`.
2. Create a subset of the dataset for training. To ensure that we are able to capture more variety in scenes and weather conditions, the dataset sampling is done such that for each object category, 10% images are selected such that same image is not selected for 2 different categories. This is done using [sample_dataset.py](sample_dataset.py) script.

### Model Training
The model training pipeline can be found in the [model_training.ipynb](model_training.ipynb) notebook. The pipeline adds some weather based augmentations on top of standard YOLO augmentations such as FLipLR, Mosaic, HSV, Grayscale, etc.

The losses used are Distribution Focal loss and Binary Cross Entropy loss.

The model training logs can be found in [results.csv](runs/detect/train/results.csv).

### Evaluation and Visualization
Model performance analysis can be found in [ModelEvaluation.md](ModelEvaluation.md)

## Future Improvements
- [ ] Include background data in model training.
- [ ] Comparative studies on different losses to improve model performance.
- [ ] Experimenting with contrastive losses such as flip contrastive loss.