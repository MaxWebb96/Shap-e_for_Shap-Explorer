from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
# from flask_socketio import SocketIO, join_room, leave_room, emit

import os
import torch

# Import the necessary modules for 3D mesh generation
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

app = Flask(__name__)
CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f'Device: {device}')

@app.route('/convert', methods=['POST'])
def convert_text_to_3d():
    data = request.json
    text = data['text']
    guidance_scale = float(data.get('guidance_scale', 16.0))  # Default to 16.0 if not provided
    generate_process = data.get('generate_process', False) 

    print(f'Text received: {text}, Guidance Scale: {guidance_scale}, Generate Process: {generate_process}')

    file_name = text + '_'+ str(int(guidance_scale)) + '.ply'
    file_name_obj = text + '_'+ str(int(guidance_scale)) + '.obj'

    local_directory = os.path.abspath('data')
    local_directory_obj = local_directory + '/obj'
    ensure_directory_exists(local_directory)

    full_path = os.path.join(local_directory, file_name)
    obj_path = os.path.join(local_directory_obj, file_name_obj)
    print(f'Full file path: {full_path}')

    if not os.path.exists(full_path):
        print('File does not exist, generating new file.')
        mesh = generate_mesh_from_text(text, guidance_scale, generate_process, local_directory)
        write_mesh_to_file(mesh, full_path)

        wrtie_mesh_to_obj(mesh, obj_path)

    return send_mesh_file(full_path)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_mesh_from_text(text, guidance_scale_get=16.0, generate_process=False, local_path=None):
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    print('Models loaded')

    # emit('progress', {'progress': 10, 'message': 'Models loaded'}, namespace='/')
    steps_to_generate = [4, 8, 16, 32, 64] if generate_process else [64]
    final_mesh = None

    for steps in steps_to_generate:
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

            karras_steps=steps,# default 64

            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        print(f'Latents sampled at {steps} steps')


        for latent in latents:
            mesh = decode_latent_mesh(xm, latent).tri_mesh()
            if steps == 64:
                final_mesh = mesh
            else:
                intermediate_file_path = os.path.join(local_path, f'{text}_{int(guidance_scale_get)}_{steps}.ply')
                print(f'Writing intermediate file to {intermediate_file_path}')
                with open(intermediate_file_path, 'wb') as f:
                    mesh.write_ply(f)
                print(f'Intermediate file written at {steps} steps')
    return final_mesh

def write_mesh_to_file(mesh, file_path):
    print(f'Writing file to {file_path}')
    with open(file_path, 'wb') as f:
        mesh.write_ply(f)

    # with open(file_path, 'w') as f:
    #     f.write_obj(f)
    print('PLY File written')

def wrtie_mesh_to_obj(mesh, file_path):
    # print(f'Writing file to {file_path}')
    if hasattr(mesh, 'export'):
        mesh.export(file_path, include_materials=True, write_texture=False, write_color=True)
        print('OBJ and MTL files written')
    else:
        print('Mesh object does not support exporting OBJ with MTL')
        with open(file_path, 'w') as f:
            mesh.write_obj(f)
        print('OBJ file written')
        process_obj_content(file_path, file_path)
        print('OBJ file processed') 
            

def process_obj_content(input_file_path, output_file_path):
    
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as file:
        for line in lines:
            if line.startswith('v '):  # Identify vertex definition lines
                parts = line.split()
                # Take only the first four elements (v x y z), ignore colors or other data
                formatted_coords = [format(float(coord), ".5f") for coord in parts[1:4]]
                processed_line = 'v ' + ' '.join(formatted_coords) + '\n'
                file.write(processed_line)
            else:
                file.write(line)  


def send_mesh_file(file_path):
    try:
        return send_file(file_path, as_attachment=True, download_name=os.path.basename(file_path) + '.ply')
    except Exception as e:
        return str(e), 500


@app.route('/get-file/<filename>', methods=['GET'])
def get_file(filename):
    print ("Requested filename:", filename)
    if filename.endswith('.ply'):
        local_directory = os.path.abspath('data')
    elif filename.endswith('.obj'):
        local_directory = os.path.abspath('data/obj')
    else:
        return jsonify({"error": "Invalid file format"}), 400
    
    file_path = os.path.join(local_directory, filename)
    if os.path.exists(file_path):
        try:
            return send_file(file_path, as_attachment=True, download_name=filename)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
    # socketio.run(app, debug=True)
