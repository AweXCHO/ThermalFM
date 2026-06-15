# 💡The repository for “Physics-collaborated Foundation Model for Thermal Infrared Imaging”

# 🔨**Installation**

Create a conda environment, activate the environment and install requirements:

```
conda create --name thermalFM python=3.9 -y
conda activate thermalFM
pip install -r requirements.txt
```

# 🚪Quick **Usage**

You can just download the weight in [https://drive.google.com/drive/folders/1hWhRLdBcqThV-UetAuCpbRxKXlzk5de3?usp=drive_link](https://drive.google.com/drive/folders/1hWhRLdBcqThV-UetAuCpbRxKXlzk5de3?usp=drive_link), then change the path in /Pre-train/ThermalFM before using ThermalFM as the encoder. 

# 📝Downstream tasks

![Results](https://github.com/AweXCHO/ThermalFM/blob/main/Pre-train/result.png)

**ALL the label and the weight can be downloaded in https://pan.baidu.com/s/1vLIbBIZE22U009sHl2zS5g code: 89xy**

## Detection

1️⃣ Download the YOLO11 code

2️⃣ Put the model located in ./Detection/model/* to ultralytics-yolo11-main/ultralytics/nn/backbone/

3️⃣ Modify the ultralytics-yolo11-main/ultralytics/nn/tasks.py

4️⃣ Download the images in *Citations.md/Detection* and labels&weights in 

5️⃣ Run the detect_*_myswin.py

## Single-Class Segmentation

1️⃣ Download the mmsegmentation

2️⃣ Put the model located in ./Single-Class Segmentation/* to mmsegmentation/

3️⃣ Modify the __init__.py in mmsegmentation/

4️⃣ Download the images in *Citations.md/Detection* and labels&weights in 

5️⃣ Run the inference of mmsegmentation