from functools import lru_cache


@lru_cache
def _load_captioner():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    name = "Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(name), BlipForConditionalGeneration.from_pretrained(name)


def generate_caption(image_path):
    import torch
    from PIL import Image

    processor, model = _load_captioner()
    with Image.open(image_path) as image:
        inputs = processor(image.convert("RGB"), return_tensors="pt")
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)
