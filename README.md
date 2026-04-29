# DLMMDD Baseline with Resnet50
## Introduction
This is a baseline model for the DLMMDD competition.

The images are all synthetic and come from 10 different sources. The dataset is perfectly balanced among sources (all contribute the same number of images). Post-processing (operators will not be disclosed) has been applied to some test images.

- Training data: 7000 quadratic images of shape (1024, 1024) or (512, 512)
- Test data: 3000 near-quadratic images of various resolutions -- 79 % are quadratic, 35 % have shape (1024, 1024). The smaller dimension is at least 80 % of the larger dimension.

## Data visualization
You can visualize the data by running the `./notebooks/data_sample_vis.ipynb` notebook.

## Results
Using Resnet50 as the base model, random split 100 images for validation, the results are as follows:

|Config|Val Loss|Val Acc|submission Acc|
|------|--------|-------|------|
|Vanilla Resnet50|0.1710|0.95|0.8273|
|Resnet50 + train additional transforms|0.0828|0.96|0.829|
|Resnet50 + train additional transforms + scheduler|0.6180|0.97|0.868 (amazing)|
|[ConvNeXt baseline](https://www.kaggle.com/code/ambrosm/dlmmdd-baseline-with-convnext)|None|None|None|

## Reference
- https://www.kaggle.com/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge
- https://www.kaggle.com/code/ambrosm/dlmmdd-baseline-with-convnext