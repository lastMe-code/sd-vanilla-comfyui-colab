import subprocess
import threading
import time
import socket
import uuid
import requests
import gradio as gr
import os
import signal

# ===============================
# CONFIG
# ===============================
COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_DIR = os.getcwd()

comfy_process = None


# ===============================
# START COMFYUI
# ===============================
def start_comfyui():
    global comfy_process
    if comfy_process:
        return

    cmd = [
        "python",
        "main.py",
        "--listen", COMFY_HOST,
        "--port", str(COMFY_PORT),
        "--dont-print-server"
    ]

    comfy_process = subprocess.Popen(
        cmd,
        cwd=COMFY_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )


def wait_for_comfyui(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((COMFY_HOST, COMFY_PORT), timeout=2):
                return True
        except:
            time.sleep(1)
    return False


# ===============================
# COMFYUI API
# ===============================
def submit_prompt(workflow):
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={
            "prompt": workflow,
            "client_id": str(uuid.uuid4())
        },
        timeout=10
    )
    r.raise_for_status()
    return r.json()["prompt_id"]


def get_history(prompt_id):
    r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def get_image(filename, subfolder, folder_type):
    r = requests.get(
        f"{COMFY_URL}/view",
        params={
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
    )
    r.raise_for_status()
    return r.content


# ===============================
# WORKFLOW
# ===============================
def build_workflow(prompt, width, height, steps, cfg, seed):
    ckpt = os.listdir("models/checkpoints")[0]

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt}
        },
        "2": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": prompt
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "blurry, low quality, distorted"
            }
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["2", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "denoise": 1
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "gradio"
            }
        }
    }


# ===============================
# GENERATE
# ===============================
def generate(prompt, width, height, steps, cfg, seed):
    try:
        workflow = build_workflow(prompt, width, height, steps, cfg, seed)
        prompt_id = submit_prompt(workflow)

        while True:
            history = get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id]["outputs"]
                for node in outputs.values():
                    if "images" in node:
                        img = node["images"][0]
                        return get_image(
                            img["filename"],
                            img["subfolder"],
                            img["type"]
                        )
            time.sleep(1)

    except Exception as e:
        return f"ERROR: {e}"


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    threading.Thread(target=start_comfyui, daemon=True).start()

    if not wait_for_comfyui():
        raise RuntimeError("ComfyUI gagal dijalankan")

    with gr.Blocks(css="""
          body { background-color: #0f0f11; color: #e5e5e5; }
         .gr-box { border-radius: 14px !important; }
    """) as demo:

        gr.Markdown(
            "# 🎨 ComfyUI – Clean Gradio UI\n"
            "**Fast · Stable · Colab Ready**"
        )

        with gr.Row():

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ✏️ Prompt")
                    prompt = gr.Textbox(
                        lines=4,
                        placeholder="Describe your image...",
                        value="a realistic portrait photo, cinematic lighting"
                    )

                with gr.Group():
                    gr.Markdown("### 🖼️ Image Size")
                    width = gr.Slider(512, 1024, 768, step=64, label="Width")
                    height = gr.Slider(512, 1024, 768, step=64, label="Height")

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    steps = gr.Slider(10, 50, 20, label="Steps")
                    cfg = gr.Slider(1, 12, 7, label="CFG Scale")
                    seed = gr.Number(value=-1, label="Seed (-1 Random)")

                generate_btn = gr.Button("🚀 Generate", variant="primary")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🖼️ Output")
                    output = gr.Image(type="pil", height=512)

            generate_btn.click(
                generate,
                inputs=[prompt, width, height, steps, cfg, seed],
                outputs=output
            )

    demo.launch(share=True)
