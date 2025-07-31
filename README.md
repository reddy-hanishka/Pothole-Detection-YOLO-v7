
# 🚧 Pothole Detection using YOLOv7

This project implements **YOLOv7** to detect **potholes** in real-time from road images and videos. It builds upon the official YOLOv7 repository with custom scripts, dataset training, ONNX/TensorRT export, and visual analysis notebooks.

---

## 📂 Project Structure

pothole-detection-yolov7/
├── main_new.py                        # Custom detection and experimentation scripts
├── detect.py                          # Inference on images or videos
├── train.py, train_aux.py             # Model training (standard and p6 models)
├── export.py                          # Export model to ONNX/TensorRT
├── yolov7/
│   ├── utils/
│   │   ├── datasets.py, general.py, activations.py, common.py, autoanchor.py
│   └── models/
│       ├── yolo.py, experimental.py, add_nms.py
├── notebooks/                         # Analysis and visualization
│   ├── compare_YOLOv7_vs_YOLOv5x6.ipynb, keypoint.ipynb, visualization.ipynb
├── scripts/
│   └── get_coco.sh                    # Download COCO dataset (optional)
├── output/                            # Prediction outputs (images/videos)
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation

---

## ⚙️ Installation

```bash
# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt


⸻

🧠 Training on Custom Pothole Dataset
	1.	Prepare your data in YOLO format.
	2.	Create a data/pothole.yaml describing your dataset:

train: path/to/train/images
val: path/to/val/images
nc: 1
names: ['pothole']


	3.	Train using:

python train.py --workers 8 --device 0 --batch-size 16 \
  --data data/pothole.yaml --img 640 640 \
  --cfg cfg/training/yolov7.yaml --weights yolov7.pt --name pothole_yolov7


⸻

🔍 Inference

Detect potholes on images or video:

# On a single image
python detect.py --weights weights/best.pt --conf 0.25 --img-size 640 --source input/image.jpg

# On a video file
python detect.py --weights weights/best.pt --conf 0.25 --img-size 640 --source input/road.mp4

# On webcam (0 is the default camera)
python detect.py --weights weights/best.pt --conf 0.25 --img-size 640 --source 0

Outputs are saved in runs/detect/.

⸻

📤 Export to ONNX / TensorRT

# Export to ONNX
python export.py --weights weights/best.pt --grid --simplify --img-size 640 640

# Export to TensorRT (requires ONNX first)
python export.py --weights weights/best.pt --grid --include-nms


⸻

📒 Notebooks
	•	compare_YOLOv7_vs_YOLOv5x6.ipynb: Side-by-side benchmark comparisons
	•	keypoint.ipynb: Pothole bounding box insights
	•	visualization.ipynb: Inference visualizations and plots

⸻

📦 Requirements

Create requirements.txt like this:

torch>=1.12.0
opencv-python
matplotlib
numpy
pandas
seaborn
scipy
tqdm
pyyaml
thop

Then run:

pip install -r requirements.txt


⸻

🖼 Sample Output


⸻

🔗 Reference

This project is based on YOLOv7. Read the paper here.
https://link.springer.com/chapter/10.1007/978-981-99-6568-7_36




Let me know if you want a shorter or more academic version!
