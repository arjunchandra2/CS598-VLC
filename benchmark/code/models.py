import torch
from libs.DAC.src.open_clip import create_model_and_transforms, tokenize
from libs.clipora.clipora.lora.inject import inject_linear_attention
from libs.clipora.clipora.lora.attention import InjectedMultiHeadAttention
from open_clip import create_model_from_pretrained, get_tokenizer, get_model_config
from peft import PeftModel, LoraConfig, get_peft_model


def get_ft_model(device):
    CHECKPOINT_PATH = "/projectnb/cs598/students/ac25/CS598-VLC/finetuning/clipora/output_r4_a8/checkpoint_val_150000"

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        "ViT-B-32",
        "openai",
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        image_mean=None,
        image_std=None,
    )

    model_config = get_model_config("ViT-B-32")

    model = inject_linear_attention(
        model=model,
        encoders={"transformer"},
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["text_cfg"]["heads"],
    )

    model = inject_linear_attention(
        model=model,
        encoders={"visual.transformer"},
        embed_dim=model_config["vision_cfg"]["width"],
        num_heads=12,
    )

    # Load LoRA-wrapped model + weights
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
    model.to(device)
    model.eval()

    def model_forward(images, text):
        image_features, text_features, logit_scale = model(images, text)
        return image_features, text_features

    return model_forward, preprocess_train, preprocess_val, tokenize



def get_DAC_SAM(device):

    CHECKPOINT_PATH = "/projectnb/cs598/projects/comp_reason/CS598-VLC/benchmark/checkpoints/sam_cp.pt"

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

    return model_forward, preprocess_train, preprocess_val, tokenize

def get_DAC_SAM_base(device):

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        "ViT-B/32",       
        "openai",    #can also change this here to load from open clip library 
        precision="amp",  
        device=device,    
        jit=False,      
        force_quick_gelu=False,
        image_mean=None,
        image_std=None,
    )

    model.eval()

    # Define the model's forward pass function
    def model_forward(images, text):
        image_features, text_features, logit_scale = model(images, text)
        return image_features, text_features

    # Return the forward pass function and preprocessing functions
    return model_forward, preprocess_train, preprocess_val, tokenize

def get_DAC_SAM_base_2(device):
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        "ViT-B/32",       
        "laion400m_e32",    #loading same base model but different pretraining data
        precision="amp",  
        device=device,    
        jit=False,      
        force_quick_gelu=False,
        image_mean=None,
        image_std=None,
    )

    model.eval()

    # Define the model's forward pass function
    def model_forward(images, text):
        image_features, text_features, logit_scale = model(images, text)
        return image_features, text_features

    # Return the forward pass function and preprocessing functions
    return model_forward, preprocess_train, preprocess_val, tokenize


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
    return model_forward, preprocess, preprocess, tokenizer_forward

MODELS = {
    "Fine-tuned": get_ft_model,
    "DAC-SAM": get_DAC_SAM,
    "DAC-SAM-base": get_DAC_SAM_base,
    "DAC-SAM-base-2": get_DAC_SAM_base_2,
    "ViT": get_ViT
}

def get_model(name, device):
    assert name in MODELS
    return MODELS[name](device)