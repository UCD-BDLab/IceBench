# Bring Your Own Model

This directory provides tools for training, evaluating, and comparing your own sea ice classification or segmentation models with our baseline models.

---

## Quick Start

1. Create your model file (see [Model Format](#model-format) below).
2. Run the comparison script:

```bash
python model_compare.py \
  --model_file path/to/your_model.py \
  --model_class YourModelClass \
  --mode both
```

The script automatically uses configuration files in the `configs` directory.

---

## Model Format

Your model should be a PyTorch module defined in a Python file. Below are templates for classification and segmentation models.

## ⚙ Configuration

The script uses configuration files from the `configs` directory:

- **`config_data.ini`**: Data paths, preprocessing options, and class definitions.
- **`config_model.ini`**: Model training parameters, loss function, and optimization settings.

No need to specify these manually; they load automatically.

---

## Command-Line Options

| Option          | Description |
|----------------|-------------|
| `--model_file` | Path to your model file |
| `--model_class` | Model class name in the file |
| `--task_type` | `classification` or `segmentation` (defaults to config) |
| `--mode` | `train`, `evaluate`, or `both` (default: `both`) |
| `--model_params` | JSON string of model parameters (optional) |
| `--output_dir` | Directory for results (default: `model_results`) |
| `--compare_with` | Paths to saved model checkpoints for comparison (optional) |

---

## Examples

### Training & Evaluating a Classification Model

```bash
python model_compare.py \
  --model_file my_models/my_classifier.py \
  --model_class MyClassifier \
  --task_type classification \
  --mode both
```

### Comparing Your Segmentation Model with Baselines

```bash
python model_compare.py \
  --model_file my_models/my_segmentation.py \
  --model_class MySegmentationModel \
  --task_type segmentation \
  --mode evaluate \
  --compare_with baseline_models/unet_best.pt baseline_models/deeplab_best.pt
```

### Passing Parameters to Your Model Constructor

```bash
python model_compare.py \
  --model_file my_models/my_model.py \
  --model_class MyModel \
  --model_params '{"in_channels": 5, "num_classes": 8}'
```

---

## 📂 Output

Each run creates a timestamped directory containing:

- 📌 Saved model checkpoints
- 📈 Training history and plots
- 🏆 Evaluation metrics
- 📊 Comparison charts (if comparing with other models)
- 📑 Copies of the configuration files used

---

## ⚡ PyTorch Lightning Integration

This script leverages **PyTorch Lightning** for:

- 🚀 Automatic GPU/CPU detection
- 📊 Progress tracking and logging
- ⏳ Early stopping
- 📌 Model checkpointing
- 📉 TensorBoard integration

---

## 🛠 Troubleshooting

🔹 **CUDA Out of Memory** → Reduce batch size in config.  
🔹 **Model Doesn't Fit Data** → Ensure input/output dimensions match dataset.  
🔹 **ImportError** → Check that required packages are installed.

---

Happy modeling! 🎉
