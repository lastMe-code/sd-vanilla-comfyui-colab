import gradio as gr
import torch
import sys
import os
from PIL import Image
import numpy as np

sys.path.append(os.getcwd())

import comfy.model_management as mm
from nodes import (
    KSampler,
    EmptyLatentImage,
    CLIPTextEncode,
    VAEDecode,
    VAEEncode,
    UNETLoader,
    CLIPLoader,
    VAELoader,
)

mm.set_vram_state(mm.VRAMState.NORMAL)

# =====================
# Load Models
# =====================
unet = UNETLoader().load_unet(
    "sd15_unet.safetensors",
    device="cuda"
)[0]

clip = CLIPLoader().load_clip(
    "clip_vit_l.safetensors",
    type="sd15"
)[0]

vae = VAELoader().load_vae(
    "vae_mse.safetensors"
)[0]

# =====================
# TXT2IMG
# =====================
def txt2img(prompt, steps, cfg, sampler, seed, width, height):
    torch.manual_seed(int(seed))

    positive = CLIPTextEncode().encode(clip, prompt)[0]
    negative = CLIPTextEncode().encode(clip, "")[0]

    latent = EmptyLatentImage().generate(width, height, batch_size=1)[0]

    samples = KSampler().sample(
        unet, seed, steps, cfg,
        sampler, "normal",
        positive, negative,
        latent, denoise=1.0
    )[0]

    image = VAEDecode().decode(vae, samples)[0]
    return image

# =====================
# IMG2IMG
# =====================
def img2img(image, prompt, denoise, steps, cfg, sampler, seed):
    torch.manual_seed(int(seed))

    image = image.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    latent = VAEEncode().encode(vae, image)[0]

    positive = CLIPTextEncode().encode(clip, prompt)[0]
    negative = CLIPTextEncode().encode(clip, "")[0]

    samples = KSampler().sample(
        unet, seed, steps, cfg,
        sampler, "normal",
        positive, negative,
        latent, denoise=denoise
    )[0]

    result = VAEDecode().decode(vae, samples)[0]
    return result

# =====================
# UI
# =====================
with gr.Blocks(title="SD Vanilla ComfyUI Colab") as demo:
    gr.Markdown("## 🎨 SD Vanilla – TXT2IMG & IMG2IMG")

    with gr.Tab("TXT2IMG"):
        prompt = gr.Textbox(label="Prompt", lines=3)

        with gr.Row():
            steps = gr.Slider(1, 60, 28, label="Steps")
            cfg = gr.Slider(1, 15, 7, label="CFG")

        with gr.Row():
            sampler = gr.Dropdown(
                ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"],
                value="euler",
                label="Sampler"
            )
            seed = gr.Number(123456, label="Seed")

        with gr.Row():
            width = gr.Slider(256, 768, 512, step=64)
            height = gr.Slider(256, 768, 512, step=64)

        gen = gr.Button("Generate")
        out = gr.Image()

        gen.click(
            txt2img,
            [prompt, steps, cfg, sampler, seed, width, height],
            out
        )

    with gr.Tab("IMG2IMG"):
        input_img = gr.Image(type="pil", label="Input Image")
        prompt2 = gr.Textbox(label="Prompt", lines=3)

        with gr.Row():
            denoise = gr.Slider(0.1, 1.0, 0.6, label="Denoise")
            steps2 = gr.Slider(1, 60, 28, label="Steps")

        with gr.Row():
            cfg2 = gr.Slider(1, 15, 7, label="CFG")
            sampler2 = gr.Dropdown(
                ["euler", "euler_ancestral", "dpmpp_2m"],
                value="euler"
            )

        seed2 = gr.Number(123456, label="Seed")

        run2 = gr.Button("Generate IMG2IMG")
        out2 = gr.Image()

        run2.click(
            img2img,
            [input_img, prompt2, denoise, steps2, cfg2, sampler2, seed2],
            out2
        )

demo.launch(share=True)
