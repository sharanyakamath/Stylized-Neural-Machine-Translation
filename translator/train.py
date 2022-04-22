# Fine-tune GPT-2
from huggingface import run_clm

config = {
    'model_name_or_path': 'gpt2',
    'train_file': 'coqa_train.json',
    'validation_file': 'coqa_validation.json',
    'text_column_name': 'story',
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-5,
    'block_size': 256,
    'max_train_samples': 1024,
    'num_train_epochs': 1,
    'do_train': True,
    'do_eval': False,
    'output_dir': './tmp',
    'overwrite_output_dir': True,
    'log_level': 'warning' # Set to `info` or `debug` for additional logging.
}

# If preferred, can use these arguments in the config instead of `train_file`  
# and `validation_file`, but sometimes Google Colab's IP gets throttled.
# 'dataset_name': 'coqa',
# 'dataset_config_name': 'default',

model_before_finetuning, model_after_finetuning = run_clm(config)
print('LM finetuning finished!')

# Preprocess the CoQA dataset into sentence pairs for evaluation.

dataset = load_dataset('json', data_files={'validation':'coqa_validation.json'})['validation']['story']
new_dataset = preprocess_coqa(dataset)

print('Preprocessing finished!')
print(f'...found {len(new_dataset)} instances.')
print(f'...sample instance: {new_dataset[0]}')

# Run evaluation.

model_before_finetuning.cuda()
model_before_finetuning.eval()
model_after_finetuning.cuda()
model_after_finetuning.eval()

print('Running evaluation...')

before_ppl = compute_perplexity(model_before_finetuning, tokenizer, new_dataset).item()
after_ppl = compute_perplexity(model_after_finetuning, tokenizer, new_dataset).item()
print(f'\n\nPerplexity before_finetune = {before_ppl:.3f}, after_finetune = {after_ppl:.3f}\n')

before_rouge = compute_rouge(model_before_finetuning, tokenizer, new_dataset)
after_rouge = compute_rouge(model_after_finetuning, tokenizer, new_dataset)
print(f'\n\nROUGE-3 before_finetune = {before_rouge:.3f}, after_finetune = {after_rouge:.3f}\n')

print('Evaluation finished!')
