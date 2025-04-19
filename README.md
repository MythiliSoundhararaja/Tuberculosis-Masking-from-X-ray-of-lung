Lung X-ray Segmentation using Dummy U-Net
This project demonstrates a basic semantic segmentation pipeline using a Dummy U-Net architecture on the Montgomery County Chest X-ray Set. The goal is to segment lungs from chest X-ray images using PyTorch.

📂 Dataset
We use the Montgomery County X-ray Set, which contains:

Chest X-ray images (.png)

Corresponding left and right lung masks

Dataset source: NIH Montgomery CXR Dataset

Make sure to upload and unzip the dataset in your Google Drive or local environment before running the code.

🔧 Features
Loads X-ray images and their corresponding left and right lung masks.

Resizes and transforms images for model input.

Uses a dummy U-Net-like PyTorch model for inference.

Visualizes:

Original image

Ground truth mask

Predicted mask overlay

🛠️ Dependencies
Make sure to install the following Python libraries:

!pip install torch torchvision matplotlib pillow


🚀 Running the Project

Unzip the dataset

Make sure your dataset is located at:

/content/drive/MyDrive/lung_data/NLM-MontgomeryCXRSet (1).zip
The code extracts it to /content/dataset.

Remove Unwanted Files

!rm -r /content/dataset/MontgomerySet/CXR_png/.ipynb_checkpoints
!rm -r /content/dataset/MontgomerySet/ManualMask/leftMask/.ipynb_checkpoints
Load Image and Masks

Each image has a left and right lung mask, which are combined into a single binary mask.

Model Architecture

python
Copy
Edit
class DummyUNet(nn.Module):
    def __init__(self):
        super(DummyUNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.conv(x))
Note: This is a placeholder model. For real 3D segmentation tasks, use advanced architectures like VNet, nnUNet, or MONAI-based pipelines on CT scans.

Visualization

The code uses matplotlib to visualize the original X-ray, ground truth mask, and model prediction overlay.

🧠 Results
Example output:

Original X-ray

True Lung Mask (Ground Truth)

Predicted Mask Overlay

The dummy model will not produce accurate results. This setup is meant for demonstrating the preprocessing and visualization pipeline.

🔄 Future Work
Integrate a real U-Net or pretrained model from MONAI.

Train on CT scan data for 3D segmentation using models like VNet or nnUNet.

Add evaluation metrics (IoU, Dice Coefficient).

📝 Citation
If you're using the Montgomery County Dataset, please cite the National Library of Medicine:

“Montgomery County X-ray Set provided by the U.S. National Library of Medicine.”

💬 Contact
For queries or collaborations, feel free to connect!
@mythilisoundhararajan@gmail.com

