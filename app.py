import subprocess
import threading
import time
import json
import random
import requests
import gradio as gr
import os

COMFY_API = "http://127.0.0.1:8188/prompt"

# =============================
# RUN COMFYUI IN BACKGROUND
# =============================
def run_comfyui():
    subprocess.Popen(
        ["python", "main.py", "--dont-print-server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

threading.Thread(target=run_comfyui, daemon=True).start()
time.sleep(10)

# =============================
# STYLE PRESETS
# =============================
STYLE_PRESETS = {
    "None": "",
    "Photorealistic": "photorealistic, ultra detailed, sharp focus, realistic lighting",
    "Cinematic": "cinematic lighting, film still, dramatic light, high contrast",
    "Digital Art": "digital painting, concept art, highly detailed, smooth shading",
    "Anime": "anime style, clean lineart, vibrant colors"
}

# =============================
# BUILD WORKFLOW
# =============================
def build_workflow(
    prompt, negative, width, height, steps, cfg,
    sampler, scheduler, seed, batch, denoise, init_image=None
):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["4", 1]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "Result",
                "images": ["8", 0]
            }
        }
    }

    return workflow

# =============================
# GENERATE FUNCTION
# =============================
def generate(
    prompt, negative, style,
    width, height, steps, cfg,
    sampler, scheduler, seed, batch,
    denoise
):
    style_prefix = STYLE_PRESETS.get(style, "")
    final_prompt = f"{style_prefix}, {prompt}" if style_prefix else prompt

    workflow = build_workflow(
        final_prompt, negative,
        width, height,
        steps, cfg,
        sampler, scheduler,
        seed, batch,
        denoise
    )

    r = requests.post(COMFY_API, json={"prompt": workflow})
    if r.status_code != 200:
        return "❌ Error connecting to ComfyUI"

    return "✅ Generation started. Check output folder."

# =============================
# GRADIO UI
# =============================
with gr.Blocks(title="SD Vanilla ComfyUI Colab") as demo:
    gr.Markdown("## 🎨 Stable Diffusion 1.5 – ComfyUI Colab UI")

    prompt = gr.Textbox(label="Prompt", lines=3)
    negative = gr.Textbox(label="Negative Prompt", lines=2)

    style = gr.Dropdown(
        choices=list(STYLE_PRESETS.keys()),
        value="None",
        label="Style Preset"
    )

    with gr.Row():
        width = gr.Slider(512, 1024, value=768, step=64, label="Width")
        height = gr.Slider(512, 1024, value=768, step=64, label="Height")

    with gr.Row():
        steps = gr.Slider(1, 50, value=20, label="Steps")
        cfg = gr.Slider(1.0, 12.0, value=7.0, step=0.5, label="CFG")

    with gr.Row():
        sampler = gr.Dropdown(
            ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"],
            value="euler",
            label="Sampler"
        )
        scheduler = gr.Dropdown(
            ["normal", "karras"],
            value="normal",
            label="Scheduler"
        )

    with gr.Row():
        seed = gr.Number(value=-1, label="Seed (-1 = random)")
        batch = gr.Slider(1, 4, value=1, step=1, label="Batch Size")

    denoise = gr.Slider(0.2, 1.0, value=1.0, step=0.05, label="Denoise")

    btn = gr.Button("🚀 Generate")
    output = gr.Textbox(label="Status")

    btn.click(
        generate,
        inputs=[
            prompt, negative, style,
            width, height, steps, cfg,
            sampler, scheduler, seed, batch, denoise
        ],
        outputs=output
    )

demo.launch(server_name="0.0.0.0", share=True)
