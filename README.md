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
