from datasets import load_dataset
import codecs
def get_data():
    dataset = load_dataset('json', data_files="x.json", field = "data")
    #with codecs.open('formatted_sh-formalen.json', 'r', 'utf-8-sig') as json_file:  
    #    dataset = json.load(json_file) 
    dataset = dataset["train"].train_test_split(test_size=0.2)
    return dataset

if __name__ == "__main__":
    print(get_data())