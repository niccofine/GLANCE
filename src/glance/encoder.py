"""RET-CLIP-PLUS visual encoder"""
import json, sys
import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from torchvision import transforms
from .config import RETCLIP_CODE, RETCLIP_CKPT

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_visual_encoder(device="cpu"):
    if str(RETCLIP_CODE) not in sys.path:
        sys.path.insert(0, str(RETCLIP_CODE))
    from RET_CLIP_PLUS.clip.model import CLIP
    cfg = RETCLIP_CODE / "RET_CLIP_PLUS" / "clip" / "model_configs"
    info = json.load(open(cfg / "ViT-B-16.json", encoding="utf-8"))
    if isinstance(info["vision_layers"], str):
        info["vision_layers"] = eval(info["vision_layers"])
    info.update(json.load(open(cfg / "RoBERTa-wwm-ext-base-chinese.json", encoding="utf-8")))
    model = CLIP(**info).to("cpu")
    ckpt = torch.load(RETCLIP_CKPT, map_location="cpu")
    sd = _remap_state_dict(ckpt)
    _, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading RET-CLIP checkpoint: {unexpected}")
    model.eval()
    v = model.visual.to(device).eval()
    for p in v.parameters():
        p.requires_grad = False
    return v, model.global_feature_mapping.to(device).eval()


def _remap_state_dict(ckpt):
    sd = {}
    for k, v in ckpt.items():
        if "bert.pooler" in k:
            continue
        k = k.replace("module.", "")
        if k.startswith("visual.ln_post."):
            sd[k] = v
            sd[k.replace("visual.ln_post.", "visual.transformer.ln_post.", 1)] = v
            continue
        sd[k] = v
    return sd


def squaremaskimage(image):
    gray = np.array(image.convert("L"))
    binary = gray > threshold_otsu(gray)
    largest = max(regionprops(label(binary)), key=lambda r: r.area)
    b = largest.bbox
    roi = image.crop((b[1], b[0], b[3], b[2]))
    s = max(roi.size)
    out = Image.new("RGB", (s, s))
    out.paste(roi, ((s - roi.size[0]) // 2, (s - roi.size[1]) // 2))
    return out


_resize = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
_norm = transforms.Normalize(CLIP_MEAN, CLIP_STD)


def clip_preprocess(path):
    img = squaremaskimage(Image.open(path).convert("RGB"))
    return _norm(_resize(img))


@torch.no_grad()
def encode_pair(vencoder, le_path, re_path, device):
    x = torch.stack([clip_preprocess(le_path), clip_preprocess(re_path)]).to(device)
    f = vencoder(x)
    return torch.cat([f[0], f[1]]).cpu().numpy()

def modeldef():
    with open(RETCLIP_CODE / "RET_CLIP_PLUS" / "clip" / "model_configs" / "ViT-B-16.json", encoding="utf-8") as f:
        modelinfo = json.load(f)
    if isinstance(modelinfo["vision_layers"], str):
        modelinfo["vision_layers"] = eval(modelinfo["vision_layers"])
    with open(RETCLIP_CODE / "RET_CLIP_PLUS" / "clip" / "model_configs" / "RoBERTa-wwm-ext-base-chinese.json", encoding="utf-8") as f:
        modelinfo.update(json.load(f))
    if str(RETCLIP_CODE) not in sys.path:
        sys.path.insert(0, str(RETCLIP_CODE))
    from RET_CLIP_PLUS.clip.model import CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIP(**modelinfo).to(device)
    ckpt = torch.load(RETCLIP_CKPT, map_location=device)
    return model, ckpt


def extrvencoder(model, ckpt):
    sd = _remap_state_dict(ckpt)
    _, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading RET-CLIP checkpoint: {unexpected}")
    model.eval(); model.global_feature_mapping.eval()
    vencoder = model.visual
    vencoder.fusion_layer = model.global_feature_mapping
    return vencoder


def extrduallatent(img_path, vencoder, fusion_layer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageL = clip_preprocess(img_path[0]).unsqueeze(0).to(device)
    imageR = clip_preprocess(img_path[1]).unsqueeze(0).to(device)
    left_feature = vencoder(imageL)
    right_feature = vencoder(imageR)
    feature = torch.cat((left_feature, right_feature), dim=1)
    if fusion_layer is not None:
        feature = fusion_layer(feature)
    return feature.cpu().numpy()
