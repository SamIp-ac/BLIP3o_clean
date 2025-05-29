
## Qwen3 + SigLIP2 + EVACLIP

This branch combines:
- **Qwen3** as the autoregressive backbone  (You can choose any size Qwen3 model, 0.6B, 1.7B, 4B, 8B, 14B, 32B)
- **SigLIP2** for image understanding vision encoder  
- **EVACLIP** for image generation vision encoder  

You can set up and run this in the same environment as the `main` branch.

### Available Training Modes
- **Image Understanding (I2T)**
- **Image Generation (T2I)**
- **Joint Training** (both tasks)


To choose different training tasks, update the dataloader in `train.py`:  
- Image generation data [https://github.com/JiuhaiChen/BLIP3o/blob/Qwen3-Siglip2/blip3o/train/train.py#L498]
- Image understanding data [https://github.com/JiuhaiChen/BLIP3o/blob/Qwen3-Siglip2/blip3o/train/train.py#L512]


Specific data type markers in the script:  
- **T2I** (Text-to-Image) [https://github.com/JiuhaiChen/BLIP3o/blob/Qwen3-Siglip2/blip3o/train/train.py#L503]
- **I2T** (Image-to-Text) [https://github.com/JiuhaiChen/BLIP3o/blob/Qwen3-Siglip2/blip3o/train/train.py#L517]


### Freezing the Backbone  
- Add `--freeze_backbone True` in the training script to freeze Qwen3 during training  
- Add `--freeze_backbone False` in the training script  to unfreeze Qwen3 during training  (we recommend unfreeze backbone when you train image understanding tasks)

### Adjust your batch size according to your GPU setup!
