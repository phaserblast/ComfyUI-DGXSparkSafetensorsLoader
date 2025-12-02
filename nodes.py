# https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader
# Copyright (c) 2025 Phaserblast. All rights reserved.
# Released under the Apache 2.0 license.
import torch
import folder_paths
import comfy

# https://github.com/foundation-model-stack/fastsafetensors
from fastsafetensors import fastsafe_open,SafeTensorsFileLoader,SingleGroup

class DGXSparkSafetensorsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The filename of the .safetensors model to load."}),
                "device": (["cuda:0"],{"default": "cuda:0", "tooltip": "The device to which the model will be copied."}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Node for loading a .safetensors file directly into memory using GPU Direct on DGX Spark."
     
    def load_model(self, model_name, device):
        dev = torch.device(device)
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        
        # fastsafetensors
        loader = SafeTensorsFileLoader(SingleGroup(), dev)
        loader.add_filenames({0: [model_path]})
        metadata = loader.meta[model_path][0].metadata
        fb = loader.copy_files_to_device()
        keys = list(fb.key_to_rank_lidx.keys())
        sd = {} # state dictionary
        for k in keys:
            sd[k] = fb.get_tensor(k)
        #fb.close() # No!
        #loader.close() # No!
        
        # Init the model to pass to ComfyUI
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            sd = temp_sd
        model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
        if model_config == None:
            fb.close()
            loader.close()
            raise RuntimeError("Couldn't load the model.")
        model_dtype = comfy.utils.weight_dtype(sd, "")
        model_config.set_inference_dtype(model_dtype, torch.bfloat16)
        model = model_config.get_model(sd, "", device=None)
        
        # Use this instead of load_model_weights()
        # I think 'diffusion_model' is a subclass of torch.nn.Module,
        # so we can use assign=True in load_state_dict which avoids
        # a copy. Just make sure you don't free the tensors read
        # by fastsafetensors if we use this option.
        #
        # The following duplicates the functionality of load_model_weights():
        #
        # Ensure stuff isn't duplicated on the GPU
        model = model.to(None)
        #
        sd = model_config.process_unet_state_dict(sd)
        # load_state_dict is from torch.nn.Module
        # If using assign=True, then don't free the tensors
        # loaded with fastsafetensors.
        model.diffusion_model.load_state_dict(sd, strict=False, assign=True)
        
        model = comfy.model_patcher.ModelPatcher(model, load_device=dev, offload_device=None)
        
        # Don't free anything, this is just here for completeness 
        #fb.close()
        #loader.close()
        return (model,)

