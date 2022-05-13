from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import json
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-51650")
f1 = open("../data/augmented_data.json", "r")
a = json.load(f1)
a = a['data']
i = 0
f2 = open("by_prime.txt", "w")
prefix = "augmented ShEn to formal En: "
for example in a:
	input_aug = prefix + example["translation"]["b_x"] + "<cat>" + example["translation"]["b_y"]
	input_ids = tokenizer.encode(input_aug, return_tensors="pt")
	output = model.generate(input_ids, max_length=128)
	t5_translated = tokenizer.decode(output[0], skip_special_tokens=True)
	print(t5_translated)
	print("\n")
	f2.write(t5_translated)
	f2.write("\n")
