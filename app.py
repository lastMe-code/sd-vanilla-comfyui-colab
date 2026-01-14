import subprocess
import threading
import time
import socket
import json
import uuid
import requests
import gradio as gr
import os
import signal

COMFYUI_DIR = os.getcwd()
COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

comfy_process = None


# ===============================
# START COMFYUI BACKEND
# ===============================
def start_comfyui():
    global comfy_process
    if comfy_process is not None:
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
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )


def wait_for_comfyui(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((COMFY_HOST, COMFY_PORT), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


# ===============================
# COMFYUI API FUNCTIONS
# ===============================
def submit_prompt(prompt_json):
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={
            "prompt": prompt_json,
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
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    }
    r = requests.get(f"{COMFY_URL}/view", params=params)
    r.raise_for_status()
    return r.content


# ===============================
# SIMPLE TEXT → IMAGE WORKFLOW
# ===============================
def build_workflow(prompt, width, height, steps, cfg, seed):
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": seed,
                "steps": steps
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": os.listdir("models/checkpoints")[0]
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": prompt
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": "blurry, low quality, distorted"
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
                "filename_prefix": "gradio",
                "images": ["8", 0]
            }
        }
    }


# ===============================
# GRADIO GENERATE FUNCTION
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
                        data = get_image(
                            img["filename"],
                            img["subfolder"],
                            img["type"]
                        )
                        return data
            time.sleep(1)

    except Exception as e:
        return f"ERROR: {str(e)}"


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    threading.Thread(target=start_comfyui, daemon=True).start()

    if not wait_for_comfyui():
        raise RuntimeError("ComfyUI gagal dijalankan")

    with gr.Blocks(css="""
body { background-color: #0f0f11; }
.gr-box { border-radius: 12px !important; }
""") as demo:

    gr.Markdown("""
    # 🎨 ComfyUI – Clean Gradio UI  
    **Stable · Colab-Friendly · No WebSocket**
    """)

    with gr.Row():
        # ================= LEFT PANEL =================
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ✏️ Prompt")
                prompt = gr.Textbox(
                    label="",
                    placeholder="Describe what you want to generate...",
                    lines=4,
                    value="a realistic portrait photo, cinematic lighting"
                )

            with gr.Group():
                gr.Markdown("### 🖼️ Image Size")
                width = gr.Slider(512, 1024, 768, step=64, label="Width")
                height = gr.Slider(512, 1024, 768, step=64, label="Height")

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                steps = gr.Slider(10, 50, 20, label="Steps")
                cfg = gr.Slider(1, 12, 7, label="CFG Scale")
                seed = gr.Number(value=-1, label="Seed (-1 = Random)")

            generate_btn = gr.Button(
                "🚀 Generate Image",
                variant="primary"
            )

        # ================= RIGHT PANEL =================
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🖼️ Output")
                output = gr.Image(
                    label="",
                    type="pil",
                    height=512
                )

    generate_btn.click(
        fn=generate,
        inputs=[prompt, width, height, steps, cfg, seed],
        outputs=output
    )

demo.launch(share=True)
