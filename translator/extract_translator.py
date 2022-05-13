from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-54096")
f = open("../data/extract.txt", "r")
a = f.readlines()
f1 = open("../data/extract_bx.txt", "w")
prefix = "translate Shakespearean English to modern English: "
print("Starting translation")
for sh in a:
	input_sh = prefix + sh
	input_ids = tokenizer.encode(input_sh, return_tensors="pt")
	output = model.generate(input_ids, max_length=128)
	t5_translated = tokenizer.decode(output[0], skip_special_tokens=True)
	print(input_sh)
	print(t5_translated)
	print("\n")
	f1.write(t5_translated)
	f1.write("\n")
f.close()
f1.close()
