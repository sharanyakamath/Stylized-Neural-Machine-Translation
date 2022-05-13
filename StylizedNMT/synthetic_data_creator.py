from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from data_utils import get_data
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    source_lang = "Shakespearean Text"
    target_lang = "Ground Truth Formalized"
    prefix = "translate Shakespearean English to modern formal English: "
    inputs = [prefix + example[source_lang] for example in examples["translations"]]
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

model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-3300")
#dataset = load_dataset('json', data_files="x.json", field = "data")
#shakespeare = dataset["train"].train_test_split(test_size=0.2)
shakesorig = open("/content/drive/MyDrive/SEM2/685/Project/Model3_sh-formalen/shakespeare_original.txt", "r")
Lines = shakesorig.readlines()
#tokenized_text = shakespeare.map(preprocess_function, batched=True)

f = open("t5_formaltranslations_unshuffled.csv", "w")
prefix = "translate Shakespearean English to modern formal English: "
i=0
for line in Lines:
  print (i)
  i=i+1
  input = prefix + line
  input_ids = tokenizer.encode(input, return_tensors="pt")
  output = model.generate(input_ids, max_length=50)
  t5_translated = tokenizer.decode(output[0], skip_special_tokens=True)
  f.write(t5_translated)
  f.write("\n")

f.close()  
