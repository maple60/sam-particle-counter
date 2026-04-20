# SAM Particle Counter

## Get Started

### Clone the repository

Clone the repository, then set up the virtual environment and install dependencies.

#### Windows

```powershell
uv sync
.venv\Scripts\activate
```

#### macOS/Linux

```bash
uv sync
source .venv/bin/activate
```

### Setup SAM2

Setup the Segment Anything Model2 (SAM2).

### Windows

```powershell
setup\setup_sam2.bat
```

### macOS/Linux

```bash
./setup/setup_sam2.sh
```

### Launch the application

```powershell
uv run main.py
```

## Quick Start Workflow

This is the typical operation flow for particle counting in Napari. The wording matches the `main.py` UI button names so users can map README steps directly to the UI.

1. **Load an image**  
   Open an image in Napari (drag and drop, or `File > Open...`). When the first image is added, the corresponding ROI layer (`<image_name>_ROI`) is created automatically.
2. **Create/select the ROI layer and draw a rectangle**  
   Select the `*_ROI` layer, then draw one rectangular ROI with the Shapes tool (if multiple ROIs exist, the most recently drawn ROI is used).
3. **Run crop**  
   Execute **`Crop to ROI`** in the right dock to create the `<image_name>_cropped` layer.
4. **Run SAM2 auto segmentation**  
   Select `*_cropped` (or a related layer) and execute **`Run SAM2 auto segmentation`** in the right dock. Adjust parameters such as `Output mode` as needed.
5. **Check particle counts and export**  
   After reviewing segmentation results, execute **`Export segmentation artifacts`** in the right dock. In the completion message, `sam2=...` and `final=...` indicate particle counts.
