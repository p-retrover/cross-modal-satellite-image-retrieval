## Compiling Manim Architecture Animations on Fedora Linux

This guide documents the complete setup, compilation workflow required to render `CrossModalArchitecture` (`animation/cross_modal_manim.py`) into a 1080p60 MP4 video locally on Fedora Linux.

---

### 1. System Dependencies (`dnf`)

Manim requires native C/C++ rendering headers (`cairo`, `pango`), `ffmpeg` for video encoding, and a targeted set of TeX Live packages to compile `MathTex` equations into DVI/SVG vector primitives.

* **Core Graphics & Video Tooling:** Install the Python build headers, Cairo/Pango vector libraries, and FFmpeg:
```bash
sudo dnf install cairo-devel pango-devel ffmpeg python3-devel gcc pkg-config

```


* **LaTeX & DVI-to-SVG Pipeline:** Install the minimal TeX Live packages required by Manim's `standalone` cropping template and `dvisvgm` converter:
```bash
sudo dnf install texlive-scheme-basic texlive-amsmath texlive-amsfonts \
    texlive-amssymb texlive-babel-english texlive-standalone \
    texlive-preview texlive-dvisvgm

```

---

### 2. Python Virtual Environment Setup

Isolate Manim Community (`v0.22.1+`) inside the project virtual environment to prevent system-wide dependency conflicts.

1. **Create and activate the environment** from the project root (`cross_modal_satellite_image_retrieval/`):
```bash
python3 -m venv .venv
source .venv/bin/activate

```


2. **Install Manim:**
```bash
pip install --upgrade pip
pip install manim

```

---

### 3. Rendering the Animation

Run the Manim CLI by passing the **relative script path** first, followed by the **exact Python `Scene` class name** (without directory prefixes).

* **Export (1080p @ 60fps):**
```bash
manim -pqh images/cross_modal_manim_2.py CrossModalArchitecture

```

* **Custom Output Directory (Optional):** To store rendered artifacts inside `animation/media/` instead of the repository root `./media/`:
```bash
manim -pqh --media_dir animation/media animation/cross_modal_manim.py CrossModalArchitecture

```

**CLI Flag Reference:**

* **`-p` (Preview):** Automatically opens the compiled `.mp4` in your default media player upon completion.
* **`-qh` / `-ql` (Quality):** Sets output resolution and framerate (`-qh` = `1920x1080` at `60fps`; `-ql` = `854x480` at `15fps`).

**Default Output Artifact Location:**

```text
media/videos/cross_modal_manim_2/1080p60/CrossModalArchitecture.mp4

```
