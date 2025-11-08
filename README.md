# SmartVision - Object Search Application

A powerful web-based application for intelligent object detection and image search using YOLOv11. SmartVision enables users to process large batches of images, detect objects using state-of-the-art deep learning models, and search through images based on detected objects with flexible filtering options.

## 🌟 Features

- **Object Detection**: Leverages YOLOv11 for accurate and fast object detection across images
- **Batch Processing**: Process entire directories of images efficiently
- **Flexible Search Modes**: 
  - OR mode: Find images containing any of the selected classes
  - AND mode: Find images containing all selected classes
- **Threshold Filtering**: Set maximum object count thresholds for precise search results
- **Interactive Visualization**: 
  - Display bounding boxes around detected objects
  - Highlight matching objects with different colors
  - Customizable grid layout (2-6 columns)
- **Metadata Management**: Automatically save and load detection metadata for quick re-searches
- **Export Options**: Download results as JSON or ZIP files containing matched images
- **GPU/CPU Support**: Optimized for both CPU and GPU inference

## 📋 Requirements

- Python 3.11+
- ultralytics
- streamlit
- opencv-python-headless
- PyYAML
- torch
- torchvision
- Pillow
- pandas
- numpy

## 🚀 Installation

### Option 1: CPU Setup

```bash
# Create virtual environment
conda create -n yolo_image_search python=3.11 -y
conda activate yolo_image_search

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 2: GPU Setup (NVIDIA CUDA)

```bash
# Create virtual environment
conda create -n yolo_image_search_gpu python=3.11 -y
conda activate yolo_image_search_gpu

# Install PyTorch with CUDA support
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Environment Management

```bash
# List all environments
conda env list

# Remove an environment
conda remove -n yolo_image_search --all

# Check GPU availability
nvidia-smi
```

## 📖 Usage

### 1. Process New Images
- Select "Process new images" option
- Enter the path to your image directory
- (Optional) Specify a custom model weights path (defaults to `yolo11m.pt`)
- Click "Start Inference" to begin detection
- Metadata will be automatically saved for future use

### 2. Load Existing Metadata
- Select "Load existing metadata" option
- Enter the path to a previously saved metadata file (JSON format)
- Click "Load Metadata" to quickly reload results without re-processing

### 3. Search for Objects
- Choose a search mode (OR/AND)
- Select one or more object classes to search for
- (Optional) Set count thresholds for each class
- Click "Search Images" to find matches

### 4. Visualize Results
- Toggle bounding boxes display
- Adjust grid columns for better viewing
- Toggle highlight matching classes to focus on relevant detections
- View results organized in a responsive grid

### 5. Export Results
- Download search results as JSON file
- Download matched images as a ZIP file

## 📁 Project Structure

```
SmartVision-Object-Search-Application/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── instruction.txt        # Setup instructions
├── yolo11m.pt            # YOLOv11 model weights
├── src/
│   ├── inference.py      # YOLO inference engine
│   ├── utils.py          # Utility functions for metadata handling
│   └── config.py         # Configuration management
├── configs/              # Configuration files
└── processed/            # Temporary output directory
    └── temp_uploaded/
```

## 🔧 Configuration

Configuration can be modified via the `src/config.py` file:
- Model confidence threshold
- Image extensions to process
- Default model path
- Output format options

## 📊 Output Format

The application generates metadata in JSON format containing:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "detections": [
      {
        "class": "person",
        "confidence": 0.95,
        "bbox": [x1, y1, x2, y2]
      }
    ],n    "class_counts": {
      "person": 1,
      "car": 2
    }
  }
]
```

## 💡 Use Cases

- **Security & Surveillance**: Search for specific objects in video frame sequences
- **Inventory Management**: Find products in warehouse images
- **Quality Control**: Detect defects or missing components in manufacturing
- **Research**: Analyze datasets with complex object distributions
- **Content Organization**: Automatically organize image libraries by detected objects

## 🎓 Model Information

- **Model**: YOLOv11 Medium (yolo11m.pt)
- **Framework**: Ultralytics
- **Input Size**: 640x640 (auto-resized)
- **Classes**: COCO dataset (80 classes including person, car, dog, cat, etc.)

## ⚡ Performance Tips

1. **GPU Usage**: Use GPU setup for 5-10x faster processing
2. **Batch Processing**: Process multiple images at once for efficiency
3. **Metadata Caching**: Save metadata for large datasets to avoid re-processing
4. **Threshold Optimization**: Use count thresholds to narrow search results

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce grid columns
- Process smaller batches of images
- Use CPU if GPU memory is limited

### Slow Processing
- Ensure GPU is properly configured (check with `nvidia-smi`)
- Use a smaller model variant if speed is critical
- Reduce image resolution before processing

### Model Not Found
- Ensure `yolo11m.pt` is in the root directory
- Download the model with: `pip install -U ultralytics && yolo model list`

## 📜 License

Apache License 2.0 - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Last Updated**: November 9, 2025
