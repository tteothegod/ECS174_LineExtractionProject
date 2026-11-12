# ECS 174 Anime Line Art: Colab & Environment Setup

This guide explains how to set up your Colab environment to run our team's code and access the shared Kaggle dataset.

## 1. Enable the Free GPU

Role 3's U-Net training and Role 4's inference time benchmarks require a GPU.

1. In the top Colab menu bar, click **Runtime** > **Change runtime type**.
2. Under "Hardware accelerator", select **T4 GPU**.
3. Click **Save**.

## 2. Mount the Shared Google Drive
All 17k image pairs are stored in our shared Google Drive to avoid hitting GitHub repository size limits. 

1. Make sure you have added the shared `ECS174_LineExtractionProject` folder to your own Google Drive (click "Add shortcut to Drive").
2. Run this snippet at the top of your Colab notebooks to mount the drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Our dataset will now be accessible to your scripts at:
`/content/drive/MyDrive/ECS174_LineExtractionProject/data/anime/`

## 3. Clone the GitHub Repository
To run our code, you need to pull the latest version of our scripts from GitHub into your Colab session. 

Run this in a cell to clone the repo and navigate into the folder:

```bash
# Clone the repository
!git clone https://github.com/tteothegod/ECS174_LineExtractionProject.git

%cd ECS174_LineExtractionProject
```

## 4. Install Shared Dependencies

To prevent version conflicts, everyone must use the libraries defined in our shared `requirements.txt`. Once you clone the GitHub repo into Colab, run this in a cell to install them:

```bash
!pip install -r requirements.txt
```

**Our core stack:**

* `torch`, `torchvision`: For Role 3's U-Net model and training loop.
* `opencv-python`, `scikit-image`: For Role 2's Canny baseline and preprocessing.
* `numpy`: For mask thresholding and matrix operations.
* `streamlit`: For Role 4's final interactive visualization demo.

