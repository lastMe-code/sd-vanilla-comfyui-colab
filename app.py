import os, random, time, re, uuid
import torch
import numpy as np
from PIL import Image
import gradio as gr

from nodes import NODE_CLASS_MAPPINGS

CheckpointLoader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()

# === LOAD MODEL ===
with torch.inference_mode():
    model, clip, vae = CheckpointLoader.load_checkpoint(
        "v1-5-pruned-emaonly.safetensors"
    )

RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)

def safe_path(prompt):
    p = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
    return os.path.join(
        RESULT_DIR, f"{p}_{uuid.uuid4().hex[:6]}.png"
    )

@torch.inference_mode()
def generate(data):
    v = data["input"]

    seed = int(v["seed"])
    if seed == 0:
        seed = random.randint(0, 2**63)

    positive = CLIPTextEncode.encode(clip, v["positive"])[0]
    negative = CLIPTextEncode.encode(clip, v["negative"])[0]

    if v["init_image"] is None:
        latent = EmptyLatentImage.generate(
            v["width"], v["height"], batch_size=1
        )[0]
        denoise = 1.0
    else:
        image = torch.from_numpy(
            np.array(v["init_image"]).astype(np.float32) / 255.0
        )[None]
        latent = VAEEncode.encode(vae, image)[0]
        denoise = v["denoise"]

    samples = KSampler.sample(
        model,
        seed,
        v["steps"],
        v["cfg"],
        v["sampler"],
        v["scheduler"],
        positive,
        negative,
        latent,
        denoise=denoise
    )[0]

    image = VAEDecode.decode(vae, samples)[0][0]
    path = safe_path(v["positive"])
    Image.fromarray((image.cpu().numpy()*255).astype(np.uint8)).save(path)

    return path, path, seed


# ================= UI =================

ASPECTS = {
    "1:1 (1024x1024)": (1024,1024),
    "16:9 (1280x720)": (1280,720),
    "9:16 (720x1280)": (720,1280),
    "4:3 (1152x864)": (1152,864),
    "3:4 (864x1152)": (864,1152),
}

CSS = ".gradio-container { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML("""
    <div style="text-align:center;margin:20px">
      <h1>SD Vanilla – ComfyUI</h1>
      <p>Stable Diffusion 1.5 • Colab Optimized</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(
                label="Positive Prompt", lines=5
            )
            negative = gr.Textbox(
                label="Negative Prompt", lines=3
            )

            with gr.Row():
                aspect = gr.Dropdown(
                    ASPECTS.keys(),
                    value="1:1 (1024x1024)",
                    label="Aspect Ratio"
                )
                seed = gr.Number(
                    value=0, label="Seed (0 = random)", precision=0
                )
                steps = gr.Slider(
                    5, 50, value=20, step=1, label="Steps"
                )

            with gr.Accordion("Advanced Settings", open=False):
                cfg = gr.Slider(
                    1, 15, value=7, step=0.5, label="CFG"
                )
                denoise = gr.Slider(
                    0.1, 1.0, value=0.7, step=0.05,
                    label="Denoise (Img2Img)"
                )
                sampler = gr.Dropdown(
                    ["euler","euler_ancestral","dpmpp_2m","dpmpp_sde"],
                    value="euler",
                    label="Sampler"
                )
                scheduler = gr.Dropdown(
                    ["normal","karras","simple"],
                    value="karras",
                    label="Scheduler"
                )
                init_image = gr.Image(
                    label="Init Image (optional)",
                    type="numpy"
                )

            run = gr.Button("🚀 Generate", variant="primary")

        with gr.Column():
            download = gr.File(label="Download Image")
            output = gr.Image(label="Result", height=480)
            used_seed = gr.Textbox(
                label="Seed Used",
                interactive=False,
                show_copy_button=True
            )

    def ui_call(
        pos, neg, aspect, seed, steps,
        cfg, denoise, sampler, scheduler, init_image
    ):
        w,h = ASPECTS[aspect]
        return generate({
            "input": {
                "positive": pos,
                "negative": neg,
                "width": w,
                "height": h,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "denoise": denoise,
                "sampler": sampler,
                "scheduler": scheduler,
                "init_image": init_image
            }
        })

    run.click(
        fn=ui_call,
        inputs=[
            positive, negative, aspect, seed,
            steps, cfg, denoise, sampler,
            scheduler, init_image
        ],
        outputs=[download, output, used_seed]
    )

demo.launch(share=True)
