from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import argparse
    
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', dest='filename', type=str, help='Name of input file')
    parser.add_argument('--outfile', dest='outfile', type=str, help='Name of output file')
    
    args = parser.parse_args()
    filename = args.filename
    outfile = args.outfile
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device is ", device)
    
    tokenizer = AutoTokenizer.from_pretrained("filco306/gpt2-shakespeare-paraphraser")
    model = AutoModelForCausalLM.from_pretrained("filco306/gpt2-shakespeare-paraphraser", pad_token_id=tokenizer.eos_token_id)
    model.to(device)
    
    base_input_path = "./shakespeare/sparknotes/merged/"
    base_output_path = "./output/"

    self.generate(base_input_path + filename, base_output_path + outfile, tokenizer, model, device)
    
    
def generate(input_path, output_path, tokenizer, model, device):
    
    input = None
    for filename in files:
        f = open(input_path, "r")
        input = f.read()
    
    output_file = open(output_path, "w")
    torch.manual_seed(0)
    sentences = re.split('\n', input)
    for sentence in sentences:
        input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
        greedy_output = model.generate(input_ids, max_length=input_ids.shape[1]*1.5, early_stopping = True, top_p=0.90)
        output_file.write('\n')
        output_file.write(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
            