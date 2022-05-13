from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import argparse
    
    
def main():
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device is ", device)
    
    tokenizer = AutoTokenizer.from_pretrained("filco306/gpt2-shakespeare-paraphraser")
    model = AutoModelForCausalLM.from_pretrained("filco306/gpt2-shakespeare-paraphraser", pad_token_id=tokenizer.eos_token_id)
    model.to(device)
    
    generate("../data/extract.txt", "extract_baseline3.txt", tokenizer, model, device)
     
def generate(input_path, output_path, tokenizer, model, device):
    
    f = open(input_path, "r")
    input = f.readlines()
    
    output_file = open(output_path, "w")
    print(input)
    for sentence in input:
        input_ids = tokenizer.encode(sentence.strip(), return_tensors='pt').to(device)
        greedy_output = model.generate(no_repeat_ngram_size=2, input_ids=input_ids, max_length=80, early_stopping = True, num_beams=5)
        output_file.write('\n')
        output_file.write(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

main()
