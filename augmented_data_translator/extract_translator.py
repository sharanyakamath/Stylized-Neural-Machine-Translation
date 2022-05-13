from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("t5-small")
#tokenizer.add_tokens("<cat>")
model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-51650")
f1 = open("../data/extract_bx.txt", "r")
f2 = open("../data/extract_by.txt", "r")
a1 = f1.readlines()
a2 = f2.readlines()
f3 = open("../data/extract_by_prime.txt", "w")
prefix = "augmented ShEn to formal En: "
print("Starting translation")
for i in range(len(a1)):
	input_sh = prefix + a1[i] + "<cat>" + a2[i]
	input_ids = tokenizer.encode(input_sh, return_tensors="pt")
	output = model.generate(input_ids, max_length=128)
	t5_translated = tokenizer.decode(output[0], skip_special_tokens=True)
	print(input_sh)
	print(t5_translated)
	print("\n")
	f3.write(t5_translated)
	f3.write("\n")
f1.close()
f2.close()
f3.close()
