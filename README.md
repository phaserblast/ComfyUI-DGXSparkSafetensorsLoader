# ComfyUI-DGXSparkSafetensorsLoader
**A ComfyUI model loader that uses the [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) library to perform a very fast, zero-copy load from storage to VRAM.**

_This is very experimental, and may destroy the universe. So please don't use it in a production environment under any circumstances._

On DGX Spark, fastsafetensors is a massive improvement over the Hugging Face safetensors library for loading AI models. The Hugging Face library doesn't work well with the DGX Spark due to its architecture and memory design. Models load very slowly and sometimes use up to 2x memory during loading. This can cause large models to exceed the Spark's RAM capacity and fail, even when the model should fit in under half of the machine's RAM capacity.

This node doesn't require ComfyUI to be launched with the --cache-none or --disable-mmap options. The default options should work fine.

Here's an example of memory usage during and after loading the 60GB FLUX.2-dev BF16 model to the GPU and the 17GB Mistral FP8 text encoder to the CPU. As you can see, model loading happens extremely fast and memory usage never goes over 60%:

![FLUX.2-dev memory usage](https://github.com/user-attachments/assets/6a3a4ff7-bc4e-47ea-99b7-d4961b505a01)

# How to Install
Clone this repository into your ComfyUI/custom_nodes folder:
```
cd ComfyUI/custom_nodes
git clone https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader.git
```
Install the fastsafetensors Python package. If you use a Python venv, remember activate it first:
```
source venv/bin/activate
pip install fastsafetensors
```
Restart ComfyUI, and search for the "DGX Spark Safetensors Loader" node. It should also be in the "loaders" category. Use this node in place of ComfyUI's built-in "Load Diffusion Model" node.

# Known Issues
* Memory management is broken, as there is no way to free the memory allocated by fastsafetensors. This is due to the custom memory management used by fastsafetensors, which bypasses ComfyUI's built-in memory management. The workaround is to just quit and restart ComfyUI to clear VRAM.

* Only minimal testing has been done on machines with discrete GPUs.
