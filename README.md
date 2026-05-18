# Plant Disease Detection Using YOLO11 Classification

This project fine-tunes a YOLO11 classification model to identify plant leaf diseases using the Kaggle **New Plant Diseases Dataset**.

Although the project name mentions tomato leaf disease detection, the selected dataset contains multiple crop species and disease classes. The dataset is organized as class folders, so this project uses **image classification** rather than object detection. The model predicts the disease class of the whole image; it does not draw bounding boxes around diseased regions.

## Dataset

Dataset: `vipoooool/new-plant-diseases-dataset` from Kaggle

The dataset is downloaded in the notebook with KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
```

The images are arranged by class folders, for example:

```text
train/
  Apple___Apple_scab/
  Tomato___Late_blight/
  Potato___healthy/
valid/
  Apple___Apple_scab/
  Tomato___Late_blight/
  Potato___healthy/
```

## Model

The notebook uses Ultralytics YOLO11 classification:

```python
YOLO("yolo11n-cls.pt")
```

`yolo11n-cls.pt` is the nano classification model. It is small, fast, and suitable for a Kaggle notebook experiment.

## Training Strategy

The notebook includes two training options:

1. Standard training with early stopping
2. Regularized fine-tuning

The regularized fine-tuning run uses:

```python
patience=5
lr0=0.001
weight_decay=0.001
dropout=0.2
cos_lr=True
save_period=1
```

Early stopping helps prevent overtraining by stopping if validation performance does not improve for 5 epochs. Ultralytics automatically saves the best checkpoint as `best.pt`.

## Evaluation Metrics

The main metrics are:

- **Top-1 accuracy**: the model's highest-confidence prediction is correct.
- **Top-5 accuracy**: the correct class appears among the five most confident predictions.
- **Confusion matrix**: shows which disease classes are confused with each other.

Very high validation accuracy can happen on this dataset because many images are clean, centered leaf images with similar train/validation conditions. Real-world field accuracy may be lower because of lighting, blur, cluttered backgrounds, and mixed symptoms.

## Experiment Report

This report is based on the regularized fine-tuning run:

```text
plant_disease_yolo11n_cls_regularized_finetune
```

The run used YOLO11 nano classification weights with early stopping, dropout, weight decay, and cosine learning rate scheduling. Training stopped after 15 epochs, which indicates that early stopping prevented unnecessary extra epochs once validation accuracy stopped improving.

### Final Metrics

| Metric | Value |
|---|---:|
| Final epoch | 15 |
| Final train loss | 0.14791 |
| Final validation loss | 0.01934 |
| Final top-1 accuracy | 99.328% |
| Final top-5 accuracy | 99.994% |
| Best top-1 accuracy | 99.397% at epoch 10 |
| Lowest validation loss | 0.01890 at epoch 14 |

### Training Curves

![Training results](assets/results.png)

The training loss decreases strongly from the first epoch and then gradually stabilizes. Validation loss also drops and remains low after the middle epochs, suggesting that the model converged quickly.

Top-1 accuracy starts high and reaches its best value around epoch 10. After that, the accuracy stays almost flat, which is exactly the situation where early stopping is useful. Top-5 accuracy is nearly perfect throughout the later epochs, meaning the correct disease class is almost always among the model's five most confident predictions.

### Confusion Matrix

![Confusion matrix](assets/confusion_matrix.png)

The confusion matrix is strongly diagonal, which means most validation images are classified correctly. Only a small number of off-diagonal marks appear, showing that the model makes relatively few class mistakes.

### Normalized Confusion Matrix

![Normalized confusion matrix](assets/confusion_matrix_normalized.png)

The normalized confusion matrix confirms that most classes are predicted correctly at a very high rate. A few light off-diagonal cells suggest occasional confusion between visually similar plant disease classes, especially where leaf symptoms can look alike.

### Interpretation

The model achieves extremely high validation accuracy. This is a strong result, but it should be interpreted carefully. The New Plant Diseases Dataset contains many clean, centered leaf images with controlled backgrounds, so validation performance can be much higher than performance on real field images.

In practical deployment, accuracy may decrease because real images can include:

- shadows and uneven lighting
- multiple leaves in one image
- complex backgrounds
- motion blur or low camera quality
- early-stage symptoms
- multiple diseases on the same leaf

For this reason, the reported validation accuracy should be treated as dataset performance, not guaranteed real-world performance.

### Output Artifacts

The key artifacts from this run are included in the `assets/` folder:

```text
assets/results.csv
assets/results.png
assets/confusion_matrix.png
assets/confusion_matrix_normalized.png
```

The final trained model from Kaggle is saved as:

```text
/kaggle/working/runs/plant_disease_yolo11n_cls_regularized_finetune/weights/best.pt
```

## How to Run

1. Open the notebook in Kaggle.
2. Enable GPU acceleration.
3. Run the setup and dataset download cells.
4. Run the dataset preparation and visualization cells.
5. Run the regularized fine-tuning cell.
6. Validate the best model.
7. Export the output zip from `/kaggle/working`.

## Limitations

- The model performs image classification, not object detection.
- It does not localize diseased regions with bounding boxes.
- Validation accuracy may be optimistic because the dataset images are clean and controlled.
- Real field images may contain multiple leaves, shadows, complex backgrounds, blur, or multiple diseases.

## Future Improvements

- Test on real field images outside the Kaggle dataset.
- Compare YOLO11n-cls with YOLO11s-cls or YOLO11m-cls.
- Add external validation data for a more realistic performance estimate.
- Use an object detection dataset if disease localization is required.
- Build a simple web app or Gradio demo for model inference.
