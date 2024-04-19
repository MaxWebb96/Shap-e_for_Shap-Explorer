from flask import Flask, request, send_file
from flask_cors import CORS
import os
import torch

# Import the necessary modules for 3D mesh generation
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/convert', methods=['POST'])
def convert_text_to_3d():
    data = request.json
    text = data['text']
    guidance_scale = float(data.get('guidance_scale', 16.0))  # Default to 16.0 if not provided
    print(f'Text received: {text}, Guidance Scale: {guidance_scale}') 

    file_name = text + '_'+ str(int(guidance_scale)) + '.ply'
    local_directory = os.path.abspath('data')
    ensure_directory_exists(local_directory)

    full_path = os.path.join(local_directory, file_name)
    print(f'Full file path: {full_path}')

    if not os.path.exists(full_path):
        print('File does not exist, generating new file.')
        mesh = generate_mesh_from_text(text, guidance_scale)
        write_mesh_to_file(mesh, full_path)

    return send_mesh_file(full_path)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_mesh_from_text(text, guidance_scale_get=16.0):
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    print('Models loaded')

    # adjust parameter: Guidance scale
    # guidance_scale = guidance_scale_get
    
    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale= guidance_scale_get,
        model_kwargs={'texts': [text]},
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,

        karras_steps=64,# default 64

        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    print('Latents sampled')

    for latent in latents:
        return decode_latent_mesh(xm, latent).tri_mesh()

def write_mesh_to_file(mesh, file_path):
    print(f'Writing file to {file_path}')
    with open(file_path, 'wb') as f:
        mesh.write_ply(f)
    print('File written')

def send_mesh_file(file_path):
    try:
        return send_file(file_path, as_attachment=True, download_name=os.path.basename(file_path) + '.ply')
    except Exception as e:
        return str(e), 500
    
def reconstruct_mesh_latent():
    return

if __name__ == '__main__':
    app.run(debug=True)
