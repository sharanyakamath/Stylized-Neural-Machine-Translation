from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from data_utils import get_data

tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    source_lang = "sh"
    target_lang = "en"
    prefix = "translate Shakespearean English to modern English: "
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    #print(inputs)
    #targets = [example[target_lang] for example in examples["translation"]]
    #model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    # print(model_inputs)
    # exit()

    #with tokenizer.as_target_tokenizer():
    #    labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs = {}
    model_inputs["inputs"] = inputs
    return model_inputs

model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-54096")
shakespeare = get_data()

tokenized_text = shakespeare.map(preprocess_function, batched=True)

f = open("t5_translations.tsv", "w")

#print(tokenized_text.shape)
print(tokenized_text["train"][0])
i = 0
for example in tokenized_text["train"]:
    i += 1
    print(i)
    #print("Sh: ", example["inputs"])
    input_ids = tokenizer.encode(example["inputs"], return_tensors="pt")
    output = model.generate(input_ids, max_length=50)
    t5_translated = tokenizer.decode(output[0], skip_special_tokens=True)
    #print("En: ", t5_translated)
    f.write(example["translation"]["sh"] + "\t" + t5_translated)
    f.write("\n")

for example in tokenized_text["test"]:
    i += 1
    print(i)
    #print("Sh: ", example["inputs"])
    input_ids = tokenizer.encode(example["inputs"], return_tensors="pt")
    output = model.generate(input_ids, max_length=50)
    t5_translated = tokenizer.decode(output[0], skip_special_tokens=True)
    #print("En: ", t5_translated)
    f.write(example["translation"]["sh"] + "\t" + t5_translated)
    f.write("\n")
