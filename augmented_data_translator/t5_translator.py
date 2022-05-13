from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from data_utils import get_data

def preprocess_function(examples):
    source_1 = "b_x"
    source_2 = "b_y"
    target_lang = "B_y"
    prefix = "augmented ShEn to formal En: "
    input = [prefix + example[source_1] + "<cat>" + example[source_2] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(input, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenizer = AutoTokenizer.from_pretrained("t5-small")
#tokenizer.add_tokens("<cat>")
shakespeare = get_data()
#print(shakespeare['train']['translation'])
for x in shakespeare['train']['translation']:
    print(x)
tokenized_text = shakespeare.map(preprocess_function, batched=True)
print(tokenized_text)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    metric_for_best_model="eval_loss",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_strategy="epoch",
    num_train_epochs=50,
    #fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_text["train"],
    eval_dataset=tokenized_text["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
