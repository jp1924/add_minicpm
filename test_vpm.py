import requests
import torch
from model.image_processing_minicpmv import MiniCPMVImageProcessor
from model.modeling_minicpmo import MiniCPMO
from PIL import Image
from torch.testing import assert_close
from transformers import (
    MiniCPMOConfig,
    MiniCPMOForConditionalGeneration,
    MiniCPMOImageProcessor,
    set_seed,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

set_seed(42)

torch.torch.use_deterministic_algorithms(True)


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
        all_pixel_values.extend(
            [i.flatten(end_dim=1).permute(1, 0) for i in pixel_values]
        )

    tgt_sizes = org_outputs["tgt_sizes"]
    tgt_sizes = [
        tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)
    ]
    tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

    max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

    all_pixel_values = torch.nn.utils.rnn.pad_sequence(
        all_pixel_values, batch_first=True, padding_value=0.0
    )
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


def get_new_vision_features(model: MiniCPMOForConditionalGeneration, images):
    new_img_processor = MiniCPMOImageProcessor()
    new_outputs = new_img_processor(
        images,
        return_tensors="pt",
        do_rescale=False,
        do_normalize=True,
    )
    new_outputs["pixel_attention_mask"] = new_outputs.pixel_attention_mask.to(
        torch.bool
    )
    vision_features = model.vpm(**new_outputs)
    return vision_features


def get_org_vision_features(model: MiniCPMO, images):
    org_img_processor = MiniCPMVImageProcessor.from_pretrained("./model")
    org_outputs = org_img_processor(images, return_tensors="pt")

    all_pixel_values = []
    for pixel_values in org_outputs["pixel_values"]:
        all_pixel_values.extend(
            [i.flatten(end_dim=1).permute(1, 0) for i in pixel_values]
        )

    tgt_sizes = org_outputs["tgt_sizes"]
    tgt_sizes = [
        tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)
    ]
    tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

    max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

    all_pixel_values = torch.nn.utils.rnn.pad_sequence(
        all_pixel_values, batch_first=True, padding_value=0.0
    )
    B, L, _ = all_pixel_values.shape
    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool)
    for i in range(B):
        patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

    org_outputs = {
        "pixel_values": all_pixel_values,
        "patch_attention_mask": patch_attn_mask.bool(),
        "tgt_sizes": tgt_sizes,
    }
    vision_features = model.vpm(**org_outputs)
    return vision_features


image_1 = Image.new("RGB", (1200, 676), (255, 255, 255))
image_2 = Image.new("RGB", (8914, 803), (255, 255, 255))

image_3 = Image.open(
    requests.get("https://picsum.photos/id/237/400/300", stream=True).raw
)
image_4 = Image.open(
    requests.get("https://picsum.photos/id/231/200/300", stream=True).raw
)


images = [image_1]


config = MiniCPMOConfig.from_json_file("/root/workspace/model/new_config.json")
config.attn_implementation = "eager"
config.vision_config.attn_implementation = "eager"
config.text_config.attn_implementation = "eager"
config.audio_config.attn_implementation = "eager"
new_model = MiniCPMOForConditionalGeneration.from_pretrained(
    "/root/workspace/model", config=config, device_map="cpu"
)
org_model = MiniCPMO.from_pretrained("./model", device_map="cpu")

for org_vpm_param, new_vpm_param in zip(
    org_model.vpm.parameters(), new_model.vpm.parameters()
):
    if not (org_vpm_param == new_vpm_param).all():
        raise ValueError(
            f"VPM parameters are not the same! {org_vpm_param} != {new_vpm_param}"
        )


new_outputs = get_new_vision_features(new_model, images)
org_outputs = get_org_vision_features(org_model, images)

new_outputs.last_hidden_state.shape == org_outputs.last_hidden_state.shape
assert_close(
    new_outputs.last_hidden_state, org_outputs.last_hidden_state, rtol=1e-3, atol=1e-3
)


new_outputs = get_new_pixel_values(images)
org_outputs = get_org_pixel_values(images)


new_hidden_states = new_model.vpm.embeddings(
    pixel_values=new_outputs["pixel_values"],
    pixel_attention_mask=new_outputs["pixel_attention_mask"].bool(),
    target_sizes=new_outputs["target_sizes"],
)
org_hidden_states = org_model.vpm.embeddings(
    pixel_values=org_outputs["pixel_values"],
    patch_attention_mask=org_outputs["pixel_attention_mask"],
    tgt_sizes=org_outputs["target_sizes"],
)

new_hidden_states == org_hidden_states
if not (new_hidden_states == org_hidden_states).all():
    raise ValueError(
        f"Hidden states are not the same! {new_hidden_states} != {org_hidden_states}"
    )

bsz = new_hidden_states.shape[0]
new_attention_mask = (
    _prepare_4d_attention_mask(
        new_outputs["pixel_attention_mask"].view(bsz, -1),
        new_hidden_states.dtype,
    )
    if not new_model.vpm.config._attn_implementation == "flash_attention_2"
    else new_outputs["pixel_attention_mask"].view(bsz, -1)
)

org_attention_mask = (
    _prepare_4d_attention_mask(
        org_outputs["pixel_attention_mask"].view(bsz, -1),
        org_hidden_states.dtype,
    )
    if not org_model.vpm.config._attn_implementation == "flash_attention_2"
    else org_outputs["pixel_attention_mask"].view(bsz, -1)
)


for org_layer, new_layer in zip(
    org_model.vpm.encoder.layers, new_model.vpm.encoder.layers
):
    new_residual = new_hidden_states
    org_residual = org_hidden_states

    layer_new_hidden_states = new_layer.layer_norm1(new_hidden_states)
    layer_org_hidden_states = org_layer.layer_norm1(org_hidden_states)

    layer_new_hidden_states, _ = new_layer.self_attn(
        hidden_states=layer_new_hidden_states,
        attention_mask=new_attention_mask,
        output_attentions=False,
    )
    layer_org_hidden_states, _ = org_layer.self_attn(
        hidden_states=layer_org_hidden_states,
        attention_mask=org_attention_mask,
        output_attentions=False,
    )
    break
    layer_new_hidden_states
# resampler
# Image.fromarray(attn_mask.transpose(1, 0)[0].bool().numpy()).save("attn_mask.png")
# Image.fromarray((attention_mask[:, 0].bool().numpy())[1]).save("attn_mask.png")

# siglip-navit embedding restore, 대충 normalize되어 있는 이미지 다시 복원해서 사진으로 저장하는 코드
# mean_std = np.array([[[0.5, 0.5, 0.5]]])

# restore_image = (((pixel_values[0].permute(1, 2, 0).numpy() * mean_std) + mean_std) * 255).astype(np.uint8)
# Image.fromarray(restore_image).save("restore_image.png")

# siglip encoder에서 막 4d mask로 변환 되었을때
# Image.fromarray((attention_mask == 0)[1].permute(1, 2, 0).numpy().astype(np.uint8).repeat(3, axis=2) * 255).save(
#     "4d-attn-mask.png"
# )

# resampleer attention
# t_query = self._repeat(q, bs)  # [64, 7, 3584]
# t_query = t_query.permute(1, 0, 2)  # [7, 64, 3584]
# t_query = t_query.view(7, 64, 128, -1)  # [7, 64, 128, 28]
# t_query = t_query.permute(1, 0, 2, 3)
# t_query.permute(0, 3, 2, 1)  # [7, 28, 128, 64]

# t_key = x.permute(1, 0, 2)  # [7, 1032, 3584]
# t_key = t_key.view(7, 1032, 128, -1)  # [7, 1032, 128, 28]
# t_key.permute(0, 3, 2, 1)  # [7, 28, 128, 1032]

# output = torch.matmul(t_query.permute(0, 3, 2, 1).transpose(-1, -2), t_key.permute(0, 3, 2, 1))

# t_query = q_scaled.view(7, -1, 64, 128)
# t_key = k.transpose(-2, -1).view(7, -1, 128, 1032)
# torch.matmul(t_query, t_key).view(-1, 64, 1032)
