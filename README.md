💡# The repository for “Physics-collaborated Foundation Model for Thermal Infrared Imaging”

🔨# **Installation**

Create a conda environment, activate the environment and install requirements:

```
conda create --name thermalFM python=3.9 -y
conda activate thermalFM
pip install -r requirements.txt
```

🚪# Quick **Usage**

You can just download the weight in [https://drive.google.com/drive/folders/1hWhRLdBcqThV-UetAuCpbRxKXlzk5de3?usp=drive_link](https://drive.google.com/drive/folders/1hWhRLdBcqThV-UetAuCpbRxKXlzk5de3?usp=drive_link), then change the path in /Pre-train/ThermalFM before using ThermalFM as the encoder. 

📝# Downstream tasks

![Results](https://github.com/AweXCHO/ThermalFM/blob/main/Pre-train/result.png)

## Detection

1️⃣ Download the YOLO11 code
2️⃣ Put the model located in ./Detection/model/* to /mnt/dqdisk/Code/ultralytics-yolo11-main/ultralytics/nn/backbone/
3️⃣ Modify the ultralytics-yolo11-main/ultralytics/nn/tasks.py
4️⃣ Download the images and weights in 
5️⃣ Run the detect_*_myswin.py