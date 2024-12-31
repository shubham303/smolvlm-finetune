import torch
from datasets import Dataset
import pandas as pd
from PIL import Image
import os
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Idefics3ForConditionalGeneration,
    TrainingArguments,
    Trainer
)

# Configuration
USE_QLORA = True  # Using QLoRA for efficient fine-tuning
MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
IMAGES_DIR = "images"  # Your images directory
DATASET_PATH = "train.tsv"  # Your TSV file path
OUTPUT_DIR = "./smolvlm-custom"

def load_custom_dataset(tsv_path, images_dir):
    # Read TSV file
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Create a function to load images
    def load_image(image_path):
        full_path = os.path.join(images_dir, image_path)
        try:
            image = Image.open(full_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return None
    
    # Create dataset dictionary
    dataset_dict = {
        'id': df['id'].tolist(),
        'question': df['question'].tolist() ,
        'answer': df['answer'].tolist(),
        'image_path': df['image_path'].tolist()
    }
    
    # Create Dataset object
    dataset = Dataset.from_dict(dataset_dict)
    
    # Add image loading
    dataset = dataset.map(
        lambda x: {'image': load_image(x['image_path'])},
        remove_columns=['image_path']
    )
    
    # Filter out examples where image loading failed
    dataset = dataset.filter(lambda x: x['image'] is not None)
    
    return dataset

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_ID)
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

# Initialize QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Initialize LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    use_dora=False,
    init_lora_weights="gaussian"
)

# Load and prepare model
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
    device_map="auto"
)
model.add_adapter(lora_config)
model.enable_adapters()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image = example["image"]
        question = example["question"]
        answer = example["answer"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

# Load dataset
dataset = load_custom_dataset(DATASET_PATH, IMAGES_DIR)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Training arguments
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    bf16=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=250
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)

# Optional: Push to Hub if desired
trainer.push_to_hub()