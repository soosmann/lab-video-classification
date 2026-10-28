# Video Classification — University Lab

University lab work on human action recognition using the **UCF-101** video dataset. The experiments cover a frame-based InceptionV3 classifier, video classification with an LSTM over extracted InceptionV3 features, and simple random/fixed-class baselines.

The original Python files, comments, experiment settings, and model logic are preserved. This layout groups the source code, dataset preparation scripts, and lab report into separate folders.

## Project layout

```text
.
├── README.md
├── requirements.txt                  # Original dependency versions
├── docs/
│   └── Hofmann_Marcel_Lab1.pdf        # Lab report
├── scripts/
│   └── preprocessing/
│       ├── 1_move_files.py            # Split raw videos into train/test folders
│       └── 2_extract_files.py         # Extract frames and create dataset metadata
├── src/
│   ├── data.py                       # Dataset loading, sampling, and visualization
│   ├── processor.py                  # Image loading and normalization
│   ├── extractor.py                  # InceptionV3 feature extractor
│   ├── models.py                     # LSTM model definition
│   ├── train_cnn.py                  # Train and fine-tune the frame classifier
│   ├── validate_cnn.py               # Spot-check predictions on test images
│   ├── extract_features.py           # Sequential feature extraction
│   ├── extract_features_multi.py     # Multiprocessing feature extraction
│   ├── train.py                      # Train the LSTM on feature sequences
│   ├── random_vs_fixed_mode.py       # Random and fixed-class baselines
│   ├── data_viz.py                   # Dataset statistics and plots
│   ├── plot_train_cnnlog.py          # Plot CNN training accuracy
│   └── plot_train_lstmlog.py         # Plot LSTM accuracy and loss
└── data/                             # Local dataset and generated artifacts
```

The dataset, extracted features, trained models, and training logs are not included. The lab report is available in [docs/Hofmann_Marcel_Lab1.pdf](docs/Hofmann_Marcel_Lab1.pdf).

## Environment

Create a Python virtual environment and install the original pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements include TensorFlow, Keras, NumPy, pandas, Matplotlib, seaborn, and tqdm. **FFmpeg** must also be installed separately and available as `ffmpeg` on your command line for frame extraction. InceptionV3 downloads its ImageNet weights on first use if they are not already cached.

The dependency versions are retained from the lab; installation and training have not been revalidated as part of the file reorganization. Use a Python version supported by the pinned dependencies.

## Prepare the dataset

Place the extracted UCF-101 videos and the official train/test split lists under `data/`:

```text
data/
├── UCF-101/
│   └── <class_name>/
│       └── <video_name>.avi
└── ucfTrainTestlist/
    ├── trainlist01.txt
    └── testlist01.txt
```

The preparation scripts default to split `01`. Run them **from inside `data/`**, in this order:

```bash
cd data
python ../scripts/preprocessing/1_move_files.py
python ../scripts/preprocessing/2_extract_files.py
cd ..
```

The first script moves the videos into `train/<class_name>/` and `test/<class_name>/`; it is intended to run once. The second extracts frames at the configured rate of 3 FPS alongside the videos and writes `data_file.csv`. The CSV has no header and contains one row per video:

```text
train_or_test,class_name,video_name_without_extension,number_of_frames
```

Create the output directories used by extraction and training:

```bash
mkdir -p data/sequences data/checkpoints data/logs
```

After preparation, the working data layout is:

```text
data/
├── data_file.csv
├── train/<class_name>/               # Videos and extracted frames
├── test/<class_name>/                # Videos and extracted frames
├── sequences/                       # Extracted feature arrays (.npy)
├── checkpoints/                     # Saved models (.keras)
└── logs/                            # CSV training logs and TensorBoard events
```

## Run the experiments

Run the following commands **from the project root**. The scripts resolve `data/` relative to the working directory, and settings are edited directly in the Python files rather than passed as command-line flags.

### Dataset overview and baselines

```bash
python src/data_viz.py
python src/random_vs_fixed_mode.py
```

The visualization shows frame counts and class distributions. The baseline script compares random guesses with always predicting the configured class (`YoYo` by default).

### Frame-based CNN

```bash
python src/train_cnn.py
```

This trains an InceptionV3 classifier initialized with ImageNet weights. It first trains the classification layers, then fine-tunes the upper Inception layers. The current experiment uses 20 classes and images resized to 299 × 299.

To spot-check predictions, first change the checkpoint path in `src/validate_cnn.py` to a model you generated, then run:

```bash
python src/validate_cnn.py
```

Keep the validation class selection consistent with the trained model.

### Feature extraction and LSTM

The feature extractor uses the ImageNet-pretrained InceptionV3 `avg_pool` output, producing a 2,048-value vector per sampled frame. It does not automatically use the fine-tuned CNN checkpoint.

Choose one extraction script:

```bash
python src/extract_features.py
# Alternative: multiprocessing, with one extractor model per worker
python src/extract_features_multi.py
```

The original defaults differ across scripts:

| Script | Sequence length | Class limit | Other settings |
| --- | --- | --- | --- |
| `extract_features.py` | 20 | 40 | Sequential extraction |
| `extract_features_multi.py` | 40 | 40 | 3 workers |
| `train.py` | 40 | 5 | Batch size 8, up to 100 epochs, load into memory |

Before training, make sure features exist for the sequence length and all classes selected in `src/train.py`. For example, the sequential extractor's length of 20 must be changed to 40 to match the current training settings. Feature filenames include the sequence length: `<video_name>-40-features.npy`.

```bash
python src/train.py
```

The LSTM processes the frame feature sequences and predicts a video class. The implemented model uses an LSTM with 2,048 units, a dense layer with 512 units, dropout, and a softmax output.

Dataset loading keeps videos with at least the configured sequence length and at most 300 extracted frames. When a class limit is set, the current code selects classes from the end of the alphabetically sorted class list.

### Training plots and TensorBoard

Set `training_log` in the relevant plotting script to one of your generated logs, then run:

```bash
python src/plot_train_cnnlog.py
python src/plot_train_lstmlog.py
tensorboard --logdir data/logs
```

The plotting scripts save PNGs in the current working directory and display the plots. Their original example log paths and the CNN plot's fixed 20% baseline are retained; adjust them to match the experiment being plotted.

## Notes on the original lab code

- Checkpoint and log paths in the validation and plotting scripts refer to local lab runs; those artifacts are not bundled here.
- `models.py` implements the LSTM model, although some original training comments mention other architectures.
- The default LSTM training path loads features into memory. The optional generator branch still uses the older `fit_generator` API and may need adaptation for the pinned Keras version.
- All Python files and `requirements.txt` are unchanged by this reorganization, including existing comments and experiment-specific choices.
