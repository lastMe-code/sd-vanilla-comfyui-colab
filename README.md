# SD Vanilla ComfyUI Colab

Stable Diffusion 1.5 (Vanilla) running on **ComfyUI backend** with a **Gradio UI frontend**.  
Optimized for **Google Colab**, **low VRAM GPUs**, and **mobile/tablet users**.

This project is inspired by NeuralFalconYT-style Colab setup, but with:
- More unlocked UI controls
- Stable SD 1.5 workflow
- Clean and simple structure

---

## ✨ Features

- ✅ Stable Diffusion 1.5 (vanilla)
- ✅ ComfyUI running in background
- ✅ Gradio web UI (mobile friendly)
- ✅ Text-to-Image
- ✅ Image-to-Image (via denoise)
- ✅ Custom resolution, steps, CFG
- ✅ Sampler & scheduler selection
- ✅ Random or fixed seed
- ✅ Safe for Google Colab GPU (T4 / L4 / P100)

---

## 🧰 Requirements

- Google Colab account
- GPU runtime enabled
- Internet connection

---

## 🚀 Installation (Google Colab)

### 1️⃣ Enable GPU
In Colab:
- Runtime → Change runtime type
- Hardware accelerator → **GPU**

---

### 2️⃣ Install ComfyUI

```bash
cd /content
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
