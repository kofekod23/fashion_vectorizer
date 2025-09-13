import os
import uuid
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, ViTModel, AutoImageProcessor
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_models():
    fclip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE).eval()
    fclip_proc = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()
    vit_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    segformer_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    segformer_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEVICE).eval()
    return fclip_model, fclip_proc, vit_model, vit_proc, segformer_proc, segformer_model

FCLIP_MODEL, FCLIP_PROC, VIT_MODEL, VIT_PROC, SEGFORMER_PROC, SEGFORMER_MODEL = load_all_models()

SEG_CATEGORIES = {4: "Upper_clothes", 5: "Skirt", 6: "Pants", 7: "Dress"}

def fclip_embed_image(img):
    inputs = FCLIP_PROC(images=img, return_tensors="pt").to(DEVICE)
    feats = FCLIP_MODEL.get_image_features(**inputs)
    return feats.squeeze(0).detach().cpu().numpy().astype("float32")

def fclip_embed_image_with_label(img, label):
    prompt = f"photo of a {label}"
    inputs = FCLIP_PROC(images=img, text=prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    feats = FCLIP_MODEL.get_image_features(**inputs)
    return feats.squeeze(0).detach().cpu().numpy().astype("float32")

def vit_embed_image(img):
    inputs = VIT_PROC(images=img, return_tensors="pt").to(DEVICE)
    out = VIT_MODEL(**inputs)
    feats = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
    return feats.squeeze(0).detach().cpu().numpy().astype("float32")

def segment_tous_les_objets(img_pil):
    img_resized = img_pil.resize((512, 512))
    inputs = SEGFORMER_PROC(images=img_resized, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        logits = SEGFORMER_MODEL(**inputs).logits
    seg = torch.nn.functional.interpolate(
        logits, size=img_pil.size[::-1], mode='bilinear', align_corners=False
    )[0]
    mask = seg.argmax(0).cpu().numpy().astype(np.uint8)
    for cat_idx, cat_name in SEG_CATEGORIES.items():
        bin_mask = (mask == cat_idx)
        if not np.any(bin_mask): continue
        coords = np.argwhere(bin_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        margin = 0.2
        h, w = mask.shape
        bbox_h, bbox_w = y1 - y0, x1 - x0
        y0 = max(int(y0 - bbox_h * margin), 0)
        y1 = min(int(y1 + bbox_h * margin), h)
        x0 = max(int(x0 - bbox_w * margin), 0)
        x1 = min(int(x1 + bbox_w * margin), w)
        crop_np = np.array(img_pil)[y0:y1, x0:x1]
        crop_mask = bin_mask[y0:y1, x0:x1].astype(np.uint8) * 255
        if crop_np.shape[:2] == crop_mask.shape:
            crop_rgba = np.dstack([crop_np, crop_mask])
            crop_pil = Image.fromarray(crop_rgba)
        else:
            crop_pil = Image.fromarray(crop_np)
        yield (cat_name, crop_pil)

def process_one_image(img, original_filename):
    """
    Traite une image de tenue -> renvoie [tenue_dict, ...vetement_dicts...]
    """
    results_for_weaviate = []

    # Entrée tenue
    tenue_id = str(uuid.uuid4())
    tenue_vec_fclip = fclip_embed_image(img)
    tenue_vec_vit = vit_embed_image(img)
    garment_cropped_ids = []
    garment_cropped_files = []
    cats_agg = []
    results_for_weaviate.append({
        "collection_name": "Tenues_v2025_c",
        "properties": {
            "tenueId": tenue_id,
            "garmentCroppedIds": garment_cropped_ids,
            "garmentCroppedFiles": garment_cropped_files,
            "catsAgg": cats_agg,
            "descAgg": "",
            "tenuePrimaryFile": original_filename,
        },
        "vectors": {
            "fclip": tenue_vec_fclip.tolist(),
            "vit": tenue_vec_vit.tolist(),
        },
        "image_data": img
    })

    # Vetements
    for cat_name, crop_pil in segment_tous_les_objets(img):
        crop_vec_fclip = fclip_embed_image_with_label(crop_pil, cat_name)
        crop_vec_vit = vit_embed_image(crop_pil)
        crop_id = str(uuid.uuid4())
        results_for_weaviate.append({
            "collection_name": "Vetements_v2025_c",
            "properties": {
                "origImageId": tenue_id,
                "image_original_file": original_filename,
                "imageCroppedId": crop_id,
                "categoryName": cat_name,
                "imageCroppedFile": f"crop_{crop_id}.png",
                "score": 1.0
            },
            "vectors": {
                "fclip": crop_vec_fclip.tolist(),
                "vit": crop_vec_vit.tolist(),
            },
            "image_data": crop_pil
        })

        # Mise à jour
        results_for_weaviate[0]['properties']['garmentCroppedIds'].append(crop_id)
        results_for_weaviate[0]['properties']['garmentCroppedFiles'].append(f"crop_{crop_id}.png")
        results_for_weaviate[0]['properties']['catsAgg'].append(cat_name)

    return results_for_weaviate
