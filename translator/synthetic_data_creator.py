from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import json
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-54096")
f = open("../data/Shakespearean-English merged Dataset.json", "r")
a = json.load(f)
a = a['data']
f1 = open("t5_translations_dedup.tsv", "w")
i = 0
f2 = open("unshuffled_bx.txt", "w")
prefix = "translate Shakespearean English to modern English: "
for example in a:
	sh = example["translation"]["sh"]
	input_sh = prefix + sh
	input_ids = tokenizer.encode(input_sh, return_tensors="pt")
	output = model.generate(input_ids, max_length=50)
	t5_translated = tokenizer.decode(output[0], skip_special_tokens=True)
	print(input_sh)
	print(t5_translated)
	print("\n")
	f1.write(example["translation"]["sh"] + "\t" + t5_translated)
	f1.write("\n")
	f2.write(t5_translated + "\n")
