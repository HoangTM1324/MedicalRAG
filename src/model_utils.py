import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def load_model_and_tokenizer(model_name, num_labels=3, inference_mode=False):
    """Load model base và tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    bnb_config = get_bnb_config()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    
    if not inference_mode:
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        
    return model, tokenizer

def load_finetuned_model(base_model_name, adapter_path, num_labels=3):
    """Load model đã merge với Adapter để đánh giá."""
    base_model, tokenizer = load_model_and_tokenizer(base_model_name, num_labels, inference_mode=True)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    return model, tokenizer
