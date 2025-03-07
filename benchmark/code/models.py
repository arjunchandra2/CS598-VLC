import torch
from libs.DAC.src.open_clip import create_model_and_transforms, tokenize
from open_clip import create_model_from_pretrained, get_tokenizer

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
    model.eval()

    def model_forward(images, text):
        image_features, text_features, logit_scale = model(images, text)
        return image_features, text_features

    return model_forward, preprocess_train, tokenize

def get_ViT(device):
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP-256')
    tokenizer = get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP-256')
    tokenizer_forward = lambda x: tokenizer(x, context_length=model.context_length)
    model.to(device)
    model.eval()
    def model_forward(images, text):
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return image_features, text_features
    return model_forward, preprocess, tokenizer_forward

MODELS = {
    "DAC-SAM": get_DAC_SAM,
    "ViT": get_ViT,
}

def get_model(name, device):
    assert name in MODELS
    return MODELS[name](device)