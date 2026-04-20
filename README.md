# SAM Particle Counter

SAM Particle Counter is a desktop workflow for counting particles from image data using SAM2-assisted segmentation and Napari-based visualization.

## Prerequisites

Before setup, confirm the following requirements.

- **Python**: `>=3.11.6` (defined in `pyproject.toml`)
- **Required tool**: [`uv`](https://docs.astral.sh/uv/) for dependency and virtual environment management
- **GPU/CUDA**:
  - GPU is **optional**. The app can run on CPU, but performance may be slower.
  - For GPU acceleration, use a CUDA-compatible NVIDIA GPU and install a CUDA-compatible PyTorch build for your environment.
  - Verify CUDA and PyTorch compatibility for your OS/driver before running setup.
- **OS notes**:
  - **Windows**: Use PowerShell commands shown below and run `.bat` setup scripts.
  - **macOS**: Use shell commands shown below and run `.sh` setup scripts.
  - **Linux**: Use shell commands shown below and run `.sh` setup scripts.

### Maintenance task (metadata consistency)

- Keep `pyproject.toml` `[project].description` aligned with the README project description (do not leave placeholder text such as `Add your description here`).

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
