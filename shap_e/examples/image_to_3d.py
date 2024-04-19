import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
from shap_e.util.image_util import load_image
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 1
guidance_scale = 3.0

max_width = 400
max_height = 300

source_dir = 'img'
save_dir = 'output_gif'
os.makedirs(save_dir, exist_ok=True)

# Loop through all files in the source directory
for image_name in os.listdir(source_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
        image_path = os.path.join(source_dir, image_name)
        print(f"Processing {image_path}...")
        
        image = load_image(image_path)
        if image.width > max_width or image.height > max_height:
            image.thumbnail((max_width, max_height))
        
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(images=[image] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        render_mode = 'nerf'
        size = 64 #128
        cameras = create_pan_cameras(size, device)

        pil_images = []
        for i, latent in enumerate(latents):
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            for img in images:
                if isinstance(img, Image.Image):
                    pil_img = img
                elif hasattr(img, 'cpu') and callable(getattr(img, 'cpu')):
                    pil_img = Image.fromarray(img.cpu().detach().numpy().astype('uint8'))
                elif isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img)
                else:
                    continue  # Skip if the image type is not supported
                pil_images.append(pil_img)

        gif_filename = os.path.splitext(image_name)[0] + ".gif"
        gif_path = os.path.join(save_dir, gif_filename)
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=100,
            loop=0
        )
        print(f"Generated {gif_path}")
