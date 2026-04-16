!pip -q install transformers torch gradio pillow

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr
import torch

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def analyze(img):
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return f"AI Description: {caption}"

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Smart AI Image Understanding"
)

demo.launch(share=True)
