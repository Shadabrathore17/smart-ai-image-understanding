from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import torch

# Load processor and model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# AI function
def analyze(img):
    inputs = processor(images=img, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=50
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return f"AI Description: {caption}"

# Custom Footer
footer = """
<div style="text-align:center; margin-top:20px; font-size:14px;">
    <hr>
    <p>Developed by <b>Shadab</b> © 2025</p>
    <p>
        Privacy Policy: Uploaded images are processed temporarily for AI caption generation 
        and are not stored permanently.
    </p>
</div>
"""

# Gradio Interface
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="AI Description"),
    title="Smart AI Image Understanding",
    description="""
Upload any image and get AI-generated caption instantly.
""",
    article=footer
)

# Run app
demo.launch()
