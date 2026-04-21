# SAM Particle Counter

[日本語版READMEはこちら / Japanese README](README_ja.md)

[final output result](examples/outputs/example_01_20260421T094932Z/final_overlay.png)

SAM Particle Counter is a desktop application for counting particles from image data using [SAM2](https://ai.meta.com/research/sam2/)-assisted segmentation and [napari](https://napari.org/stable/)-based visualization.

## Prerequisites

Before setup, confirm the following requirements.

- **Python**: `>=3.11.6` (defined in `pyproject.toml`)
- **Required tool**: [`uv`](https://docs.astral.sh/uv/) for dependency and virtual environment management
- **GPU/CUDA**:
  - GPU is **optional**. The app can run on CPU, but performance may be slower.
  - For GPU acceleration, use a [CUDA](https://developer.nvidia.com/cuda/toolkit)-compatible NVIDIA GPU and install a CUDA-compatible [PyTorch](https://pytorch.org/) build for your environment.
  - Verify CUDA and PyTorch compatibility for your OS/driver before running setup.
- **OS notes**:
  - **Windows**: Use PowerShell commands shown below and run `.bat` setup scripts.
  - **macOS**: Use shell commands shown below and run `.sh` setup scripts.
  - **Linux**: Use shell commands shown below and run `.sh` setup scripts.

## Get Started

### Clone the repository

Clone the repository, then set up the virtual environment and install dependencies.

```bash
uv sync
```

By default, `uv sync` resolves `torch` from PyPI. On some platforms (for example Linux x86_64), the PyPI wheel may already include CUDA runtime dependencies, while other platforms/environments may get CPU-only builds. 

Always verify your installed build first, then switch to a specific CPU/CUDA index only when needed (see examples below).

#### Switch PyTorch build (CPU / CUDA)

You can keep the same project and swap PyTorch builds to match your PC.

- Confirm current build:

```bash
uv run python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"
```

- Reinstall CPU build explicitly:

```bash
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

- Reinstall CUDA 12.1 build (example):

```bash
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

> Note 1: `--upgrade` replaces the currently installed `torch*` packages in the same `uv` environment, so running this after `uv sync` is meaningful.
>
> Note 2: You can check the driver-supported CUDA runtime with `nvidia-smi` (`CUDA Version: ...`), then select a compatible PyTorch CUDA wheel (for example `cu121`).

You can find the appropriate PyTorch installation command [here](https://pytorch.org/get-started/locally/).
Use `uv pip` instead of `pip3`.

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

This is the typical operation flow for particle counting in [napari](https://napari.org/stable/). The wording matches the `main.py` UI button names so users can map README steps directly to the UI.

1. **Load an image**  
   Open an image in [napari](https://napari.org/stable/) (drag and drop, or `File > Open...`). When the first image is added, the corresponding ROI layer (`<image_name>_ROI`) is created automatically.
2. **Create/select the ROI layer and draw a rectangle**  
   Select the `*_ROI` layer, then draw one rectangular ROI with the Shapes tool (if multiple ROIs exist, the most recently drawn ROI is used).
3. **Run crop**  
   Execute **`Crop to ROI`** in the right dock to create the `<image_name>_cropped` layer.
4. **Run SAM2 auto segmentation**  
   Select the `<image_name>_cropped` image layer, then execute **`Run SAM2 auto segmentation`** in the right dock. Adjust parameters such as `Output mode` as needed.
5. **Check particle counts and export**  
   After reviewing segmentation results, execute **`Export segmentation artifacts`** in the right dock. In the completion message, `sam2=...` and `final=...` indicate particle counts.

## Acknowledgements

This project depends heavily on many open-source software projects.
In particular, I would like to thank the developers and contributors of [napari](https://napari.org/stable/), [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), and [pandas](https://pandas.pydata.org/).

## License

This project is licensed under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-clause). 
See [LICENSE](LICENSE) for details.
