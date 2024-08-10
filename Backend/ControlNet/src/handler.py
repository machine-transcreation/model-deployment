import cv2
import einops
import numpy as np
import torch
import random
import runpod
import base64
from io import BytesIO
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# Hough
from annotator.mlsd import MLSDdetector
from hough2image import load_mlsd, mlsd_process

# HED
from annotator.hed import HEDdetector
from hed2image import load_hed, hed_process

# Depth
from annotator.midas import MidasDetector
from depth2image import load_midas, midas_process

# Canny
from annotator.canny import CannyDetector
from canny2image import load_canny, canny_process

# Load all the models
def load_models():

    global apply_mlsd, model_mlsd, ddim_sampler_mlsd
    global apply_hed, model_hed, ddim_sampler_hed
    global apply_midas, model_midas, ddim_sampler_midas
    global apply_canny, model_canny, ddim_sampler_canny
    global models_loaded

    apply_mlsd, model_mlsd, ddim_sampler_mlsd = load_mlsd()
    apply_hed, model_hed, ddim_sampler_hed = load_hed()
    apply_midas, model_midas, ddim_sampler_midas = load_midas()
    apply_canny, model_canny, ddim_sampler_canny = load_canny()
    models_loaded=True

def generate_model_parameters(model_type, input_image, prompt, params_dict):
    # default values
    defaults = {
        'common': {
            'a_prompt': 'best quality, extremely detailed',
            'n_prompt': 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
            'num_samples': 1,
            'image_resolution': 512,
            'ddim_steps': 20,
            'guess_mode': False,
            'strength': 1.0,
            'scale': 9.0,
            'seed': -1,
            'eta': 0.0
        },
        'mlsd': {
            'detect_resolution': 512,
            'value_threshold': 0.1,
            'distance_threshold': 0.1
        },
        'depth': {
            'detect_resolution': 384
        },
        'canny': {
            'low_threshold': 100,
            'high_threshold': 200
        },
        'hed': {
            'detect_resolution': 512
        }
    }

    # common and model-specific default values
    common_defaults = defaults['common']
    model_defaults = defaults.get(model_type, {})

    # output with common parameters
    output = [
        input_image,
        prompt,
        params_dict.get('a_prompt', common_defaults['a_prompt']),
        params_dict.get('n_prompt', common_defaults['n_prompt']),
        params_dict.get('num_samples', common_defaults['num_samples']),
        params_dict.get('image_resolution', common_defaults['image_resolution']),
    ]

    # model-specific parameters
    if model_type == 'mlsd':
        output += [
            params_dict.get('detect_resolution', model_defaults.get('detect_resolution', 512)),
            params_dict.get('ddim_steps', common_defaults['ddim_steps']),
            params_dict.get('guess_mode', common_defaults['guess_mode']),
            params_dict.get('strength', common_defaults['strength']),
            params_dict.get('scale', common_defaults['scale']),
            params_dict.get('seed', common_defaults['seed']),
            params_dict.get('eta', common_defaults['eta']),
            params_dict.get('value_threshold', model_defaults.get('value_threshold', 0.1)),
            params_dict.get('distance_threshold', model_defaults.get('distance_threshold', 0.1))
        ]
    elif model_type == 'depth':
        output += [
            params_dict.get('detect_resolution', model_defaults.get('detect_resolution', 384)),
            params_dict.get('ddim_steps', common_defaults['ddim_steps']),
            params_dict.get('guess_mode', common_defaults['guess_mode']),
            params_dict.get('strength', common_defaults['strength']),
            params_dict.get('scale', common_defaults['scale']),
            params_dict.get('seed', common_defaults['seed']),
            params_dict.get('eta', common_defaults['eta'])
        ]
    elif model_type == 'canny':
        output += [
            params_dict.get('ddim_steps', common_defaults['ddim_steps']),
            params_dict.get('guess_mode', common_defaults['guess_mode']),
            params_dict.get('strength', common_defaults['strength']),
            params_dict.get('scale', common_defaults['scale']),
            params_dict.get('seed', common_defaults['seed']),
            params_dict.get('eta', common_defaults['eta']),
            params_dict.get('low_threshold', model_defaults.get('low_threshold', 100)),
            params_dict.get('high_threshold', model_defaults.get('high_threshold', 200))
        ]
    elif model_type == 'hed':
        output += [
            params_dict.get('detect_resolution', model_defaults.get('detect_resolution', 512)),
            params_dict.get('ddim_steps', common_defaults['ddim_steps']),
            params_dict.get('guess_mode', common_defaults['guess_mode']),
            params_dict.get('strength', common_defaults['strength']),
            params_dict.get('scale', common_defaults['scale']),
            params_dict.get('seed', common_defaults['seed']),
            params_dict.get('eta', common_defaults['eta'])
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return output

def base64_to_numpy(base64_str):
    
    if base64_str.startswith('data:image/'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    
    image = Image.open(BytesIO(image_data))
    
    return np.array(image)



def numpy_to_base64(np_array, image_format='PNG'):
    
    image = Image.fromarray(np_array)

    buffer = BytesIO()
    image.save(buffer, format=image_format)
    byte_data = buffer.getvalue()
    
    base64_str = base64.b64encode(byte_data).decode('utf-8')    
    
    return base64_str


# HANDLER FUNCTION

def handler(job):
    
    job_input = job["input"]
    model_params = generate_model_parameters(job_input['model_type'], base64_to_numpy(job_input['input_image']), job_input['prompt'], job_input['params_dict'])
    
    model_type = job_input['model_type']

    if model_type == 'mlsd':
        output = mlsd_process(*model_params, apply_mlsd, model_mlsd, ddim_sampler_mlsd)

    elif model_type == 'depth':
        output = midas_process(*model_params, apply_midas, model_midas, ddim_sampler_midas)

    elif model_type == 'canny':
        output = canny_process(*model_params, apply_canny, model_canny, ddim_sampler_canny)

    elif model_type == 'hed':
        output = hed_process(*model_params, apply_hed, model_hed, ddim_sampler_hed)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    output_base64 = numpy_to_base64(output)
    return {"image": output_base64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})