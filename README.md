# SD Vanilla ComfyUI Colab 🚀

![Stable Diffusion](https://img.shields.io/badge/Stable_Diffusion-1.5-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Latest-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![Colab](https://img.shields.io/badge/Google-Colab-yellow)

Stable Diffusion 1.5 **Vanilla** using **ComfyUI Nodes API** + **Gradio UI**  
Optimized for **Google Colab** and **Mobile / Tablet users**.

✅ txt2img  
✅ img2img  
✅ No checkpoint loader  
✅ No VRAM hacks  
✅ Compatible with latest ComfyUI  

---

## ✨ Features

- Stable Diffusion 1.5 (vanilla)
- Text-to-Image
- Image-to-Image
- Gradio Web UI
- Mobile friendly
- Safe for Colab T4 / L4
- NeuralFalcon-style node loading

---

## 🚀 One-Click Colab

Open directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/USERNAME/sd-vanilla-comfyui-colab/blob/main/colab.ipynb
)

> `lastMe-code`

---

## 🧱 Manual Installation (Colab)

### CELL 1 — Install & Models
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
wget https://raw.githubusercontent.com/lastMe-code/sd-vanilla-comfyui-colab/main/app.py
pip install -r requirements.txt
