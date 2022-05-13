from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from data_utils import get_data


def preprocess_function(examples):
    source_lang = "Shakespearean Text"
    target_lang = "Ground Truth Formalized"
    prefix = "translate Shakespearean English to modern formal English: "
    inputs = [prefix + example[source_lang] for example in examples["translations"]]
    targets = [example[target_lang] for example in examples["translations"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenizer = AutoTokenizer.from_pretrained("t5-small")
#shakespeare = get_data()
dataset = load_dataset('json', data_files="x.json", field = "data")
shakespeare = dataset["train"].train_test_split(test_size=0.2)
tokenized_text = shakespeare.map(preprocess_function, batched=True)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=50,
    fp16=True,
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