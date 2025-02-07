dependencies = ['torch', 'torchvision']
from vision_transformer import vit_base, vit_small, vit_tiny
from utils import load_pretrained_weights
import urllib.request
from pathlib import Path

# resnet18 is the name of entrypoint
def vit_small_patch16_224_dora_wt_venice_ep100(pretrained=True, **kwargs):
    import torch
    URL = "https://huggingface.co/dgcnz/DoRA/resolve/main/vit_small_patch16_224.dora.wt_venice.ep100.pth"
    CKPT_PATH = URL.split("/")[-1]

    model = vit_small()
    urllib.request.urlretrieve(URL, CKPT_PATH)
    load_pretrained_weights(model, CKPT_PATH, "teacher", "vit_small", 16)
    # model = _resnet18(pretrained=pretrained, **kwargs)
    return model