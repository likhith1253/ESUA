from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import sys

# Disable internal warning if needed, but usually fine.
import warnings
warnings.filterwarnings("ignore")

def main():
    print("Loading model... (this may take a minute the first time)")
    
    # Load model and processor
    # We use the CPU by default as requested.
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Image URL - using a stable example
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    
    print(f"Downloading image from {img_url}...")
    try:
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    except Exception as e:
        print(f"Error downloading image: {e}")
        return

    print("Running inference...")
    # Prepare inputs
    inputs = processor(raw_image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    print("\n--- Result ---")
    print(f"Caption: {caption}")
    print("--------------")

if __name__ == "__main__":
    main()
