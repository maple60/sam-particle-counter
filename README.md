[![DOI](https://zenodo.org/badge/1215439992.svg)](https://doi.org/10.5281/zenodo.19678862)

# SAM Particle Counter

[日本語版READMEはこちら / Japanese README](README_ja.md)

<p align="center">
  <img src="examples/outputs/example_01_20260421T094932Z/cropped_image.png" width="48%" />
  <img src="examples/outputs/example_01_20260421T094932Z/final_overlay.png" width="48%" />
</p>

<p align="center">
  Left: original / Right: final overlay
</p>

SAM Particle Counter is a desktop application for counting particles from image data using [SAM2](https://ai.meta.com/research/sam2/)-assisted segmentation and [napari](https://napari.org/stable/)-based visualization.

[Online manual is available here](https://maple60.github.io/sam-particle-counter/)

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

The SAM2 setup scripts below run `uv sync`, then automatically reinstall `torch` with uv's PyTorch backend detection:

```bash
uv pip install --upgrade torch --torch-backend=auto
```

This asks uv to detect the installed GPU/CUDA driver and choose a compatible PyTorch backend. If no supported GPU backend is found, uv falls back to the CPU build.

If you install dependencies manually with `uv sync`, run the command above afterward, then verify the installed build:

```bash
uv run --no-sync python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"
```

#### Switch PyTorch build (CPU / CUDA)

If automatic detection does not choose the build you want, you can keep the same project environment and swap PyTorch builds explicitly.

- Auto-select again:

```bash
uv pip install --upgrade torch --torch-backend=auto
```

- Reinstall CPU build explicitly:

```bash
uv pip install --upgrade torch --torch-backend=cpu
```

- Reinstall CUDA 12.8 build (example):

```bash
uv pip install --upgrade torch --torch-backend=cu128
```

> Note 1: `--upgrade` replaces the currently installed `torch` package in the same `uv` environment, so running this after `uv sync` is meaningful.
>
> Note 2: You can check the driver-supported CUDA runtime with `nvidia-smi` (`CUDA Version: ...`), then select a compatible uv PyTorch backend such as `cu126` or `cu128`.
>
> Note 3: After switching PyTorch builds with `uv pip`, use `uv run --no-sync ...` to avoid replacing the selected build during project sync.

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
uv run --no-sync main.py
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

## How to cite

If you use SAM Particle Counter in your research, please cite the software release:

Kaede Konrai. (2026). maple60/sam-particle-counter: v0.1.0 (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.19678863

You can also find citation metadata in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

This project depends heavily on many open-source software projects.
In particular, I would like to thank the developers and contributors of [napari](https://napari.org/stable/), [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [grasbey](https://glasbey.readthedocs.io/en/latest/creating_palettes.html), and [pandas](https://pandas.pydata.org/).

This software was also inspired in part by existing plant image analysis workflows and tools, including *Samplify* ([Bente et al., 2026](https://doi.org/10.1111/nph.70979)).

## AI Assistant

AI tools, including GitHub Copilot and OpenAI tools (ChatGPT/Codex), were used to assist in drafting code and revising documentation.

All methodological decisions and validations were conducted by the author. 
The author assumes full responsibility for the scientific correctness and reproducibility of this software.

## License

This project is licensed under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-clause). 
See [LICENSE](LICENSE) for details.
