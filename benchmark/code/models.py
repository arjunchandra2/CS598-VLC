import torch
from libs.DAC.src.open_clip import create_model_and_transforms, tokenize

def get_DAC_SAM(device):

    CHECKPOINT_PATH = "checkpoints/sam_cp.pt"

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        "ViT-B/32",
        "openai",
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        lora=4,
        freeze_img=False,
        kqv_lora=False,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith(
        "module"
    ):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    return model, preprocess_train, tokenize

MODELS = {
    "DAC-SAM": get_DAC_SAM,
}

def get_model(name, device):
    assert name in MODELS
    return MODELS[name](device)