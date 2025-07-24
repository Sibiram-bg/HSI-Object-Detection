# HSI-Object-Detection-Final

This is a corrected and enhanced version of the HSI-Object-Detection-NPU project. The original code has been updated to work with modern Python libraries (Python 3.12, latest PyTorch), and numerous bugs have been fixed to ensure it runs smoothly on a local machine.

This repository contains two complete, independent training experiments:
1.  **Training from Scratch:** The original methodology with added data augmentation.
2.  **Transfer Learning:** A more advanced strategy using a pre-trained VGG-16 model from ImageNet to achieve higher accuracy.

**Final mAP Score (Training from Scratch): 76.0%**

---
### Setup & Usage

**1. Environment Setup**
```bash
# Install required libraries
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install numpy tifffile tqdm
```

**2. Data Preparation**
- Download the HSI dataset from one of the official links below. The `.zip` file contains both the HSI and RGB data. Unzip it and place the `VOC2007` folder inside this project directory.
    - **Baidu Cloud:** [[link](https://pan.baidu.com/s/1mtXDJfU6M8F60GZinLam-w), password: 6shr]
    - **OneDrive:** [[link](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yanlongbin_mail_nwpu_edu_cn/ERsB07TPh8RGrNpsgIejn38B0rmwzJEBgLmL5hzwvYlV7g?e=Upk6iW)]
- Edit `create_data_lists.py` to point `voc07_path` to your `VOC2007` folder.
- Run the script once to generate the necessary JSON files:
  ```bash
  python create_data_lists.py
  ```

**3. Running the Experiments**

#### Experiment 1: Training from Scratch
- **Train:** `python train.py --batch_size 8`
- **Evaluate:** `python eval.py`
- **Detect:** `python detect.py --dataset_path "path/to/your/VOC2007"`
- **Results:** Saved in the `results/` folder.

#### Experiment 2: Training with Transfer Learning
- **Train:** `python train_transfer_learning.py --batch_size 8`
- **Evaluate:** `eval_transfer.py`
- **Detect:** `detect_transfer.py --dataset_path "path/to/your/VOC2007"`
- **Results:** Saved in the `results_transfer/` folder.

---
### Key Bug Fixes & Improvements

- **Library Modernization:** Migrated from `libtiff` to `tifffile` for modern Python compatibility.
- **PyTorch Compatibility:** Fixed `torch.load` errors (`_pickle.UnpicklingError`) by adding `weights_only=False` and resolved `torch.uint8` indexing warnings.
- **System & Hardware Fixes:** Solved Windows `DataLoader` freezes by setting `workers=0`. Corrected hardcoded GPU IDs and helped determine optimal `batch_size` to prevent memory errors.
- **Workflow & Visualization:**
    - Implemented a robust transfer learning strategy using pre-trained ImageNet weights.
    - Created a fully separate set of scripts to manage the two experiments cleanly.
    - Fixed file path bugs in `detect.py` by using a command-line argument for the dataset path.
    - Modified `detect.py` to work without separate RGB images by generating false-color composites from `.tiff` files.
    - Improved detection visualizations with clearer text labels.

### Acknowledgements

This project is a modified and enhanced version of the original [HSI-Object-Detection-NPU](https://github.com/yanlongbinluck/HSI-Object-Detection-NPU) repository. The original paper, "Object Detection in Hyperspectral Images," can be found [here](https://ieeexplore.ieee.org/document/9365545).
