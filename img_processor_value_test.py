import requests
import torch
from model.image_processing_minicpmv import MiniCPMVImageProcessor
from PIL import Image

from transformers.models.minicpm_o import MiniCPMOImageProcessor


def get_new_pixel_values(images):
    new_img_processor = MiniCPMOImageProcessor()
    new_outputs = new_img_processor(
        images,
        return_tensors="pt",
        do_rescale=False,
        do_normalize=True,
    )
    return new_outputs


def get_org_pixel_values(images):
    org_img_processor = MiniCPMVImageProcessor.from_pretrained("./model")
    org_outputs = org_img_processor(images, return_tensors="pt")

    all_pixel_values = []
    for pixel_values in org_outputs["pixel_values"]:
        all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

    tgt_sizes = org_outputs["tgt_sizes"]
    tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
    tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

    max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

    all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
    B, L, _ = all_pixel_values.shape
    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool)
    for i in range(B):
        patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

    return {
        "pixel_values": all_pixel_values,
        "pixel_attention_mask": patch_attn_mask,
        "target_sizes": tgt_sizes,
    }


image_1 = Image.new("RGB", (1200, 676), (255, 255, 255))
image_2 = Image.new("RGB", (8914, 803), (255, 255, 255))

image_3 = Image.open(requests.get("https://picsum.photos/id/237/400/300", stream=True).raw)
image_4 = Image.open(requests.get("https://picsum.photos/id/231/200/300", stream=True).raw)

images = [image_1, image_2, image_3, image_4]
new_outputs = get_new_pixel_values(images)
org_outputs = get_org_pixel_values(images)

new_outputs["pixel_values"] == org_outputs["pixel_values"]

if not (new_outputs["pixel_values"] == org_outputs["pixel_values"]).all():
    print("Pixel values are not the same!")
else:
    print("Pixel values are the same!")
