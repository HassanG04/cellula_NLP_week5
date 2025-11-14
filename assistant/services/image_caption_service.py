from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model once (important for performance)
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

def generate_caption(image_path: str) -> str:
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
