import os, random, uuid, time
import torch
import numpy as np
from PIL import Image
import gradio as gr

from nodes import NODE_CLASS_MAPPINGS


# =========================
# Load Nodes
# =========================
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()


# =========================
# Load SD 1.5
# =========================
with torch.inference_mode():
    unet = UNETLoader.load_unet("sd15-unet.safetensors", "default")[0]
    clip = CLIPLoader.load_clip("clip-vit-l14.safetensors")[0]
    vae = VAELoader.load_vae("sd15-vae.safetensors")[0]


# =========================
# Output
# =========================
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def save_image(img):
    name = f"{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(OUT_DIR, name)
    Image.fromarray(img).save(path)
    return path


# =========================
# Core Generate
# =========================
@torch.inference_mode()
def generate(
    prompt,
    negative,
    width,
    height,
    steps,
    cfg,
    denoise,
    seed,
    init_image=None
):
    if seed == 0:
        seed = random.randint(0, 2**63)

    pos = CLIPTextEncode.encode(clip, prompt)[0]
    neg = CLIPTextEncode.encode(clip, negative)[0]

    if init_image is None:
        latent = EmptyLatentImage.generate(width, height)[0]
    else:
        img = torch.from_numpy(init_image).float() / 255.0
        img = img.unsqueeze(0)
        latent = VAEEncode.encode(vae, img)[0]

    samples = KSampler.sample(
        unet,
        seed,
        steps,
        cfg,
        "euler",
        "simple",
        pos,
        neg,
        latent,
        denoise=denoise
    )[0]

    decoded = VAEDecode.decode(vae, samples)[0]
    img = (decoded[0].cpu().numpy() * 255).astype(np.uint8)

    return save_image(img), seed


# =========================
# Gradio
# =========================
def ui_generate(
    prompt,
    negative,
    image,
    steps,
    cfg,
    denoise,
    seed
):
    init = None
    if image is not None:
        init = np.array(image)

    path, used_seed = generate(
        prompt,
        negative,
        512,
        512,
        steps,
        cfg,
        denoise,
        seed,
        init
    )

    return path, path, used_seed


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Stable Diffusion 1.5 (txt2img + img2img)")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox("a cinematic portrait, ultra detailed", label="Prompt")
            negative = gr.Textbox("low quality, blurry", label="Negative")
            image = gr.Image(type="numpy", label="Image (optional img2img)")

            steps = gr.Slider(1, 50, 20, step=1, label="Steps")
            cfg = gr.Slider(1, 15, 7, step=0.5, label="CFG")
            denoise = gr.Slider(0.1, 1.0, 1.0, step=0.05, label="Denoise")
            seed = gr.Number(0, precision=0, label="Seed (0=random)")

            btn = gr.Button("🚀 Generate")

        with gr.Column():
            file = gr.File(label="Download")
            img = gr.Image()
            used_seed = gr.Textbox(label="Seed Used")

    btn.click(
        ui_generate,
        [prompt, negative, image, steps, cfg, denoise, seed],
        [file, img, used_seed]
    )

demo.launch(share=True)
