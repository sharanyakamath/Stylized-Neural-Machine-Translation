from datasets import load_dataset

def get_data():
    dataset = load_dataset('json', data_files="../data/augmented_data.json", field = "data")
    dataset = dataset["train"].train_test_split(test_size=0.2)
    return dataset

if __name__ == "__main__":
    print(get_data())
