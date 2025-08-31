# required
pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
revision = None

# optional
dataset_name = "" 
train_data_path = "" # kcd dataset path

caption_column = "text"  
output_dir = "outputs/kodi"  
seed = 42  
resolution = 512  
train_batch_size = 32  
num_train_epochs = 50  
learning_rate = 1e-4  
lr_scheduler = "constant"
lr_warmup_steps = 0 
checkpointing_steps = 5000
random_flip = True
dataloader_num_workers = 4

variant = None
dataset_config_name = None
train_data_dir = None
image_column = "image"
max_train_samples = None
center_crop = False
max_train_steps = None
gradient_accumulation_steps = 4
snr_gamma = None
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08
max_grad_norm = 1.0
prediction_type = None
logging_dir = "logs"
mixed_precision = None  # ["no", "fp16", "bf16"]
checkpoints_total_limit = None
resume_from_checkpoint = None
noise_offset = 0
rank = 4
