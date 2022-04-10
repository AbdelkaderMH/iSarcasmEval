import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer



class TrainDataset(Dataset):
    def __init__(self, df, pretraine_path='xlm-roberta-base', max_length=128):
        self.df = df
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    def __getitem__(self, index):
        text = self.df.iloc[index]['tweet']
        label = self.df.iloc[index]["sarcastic"]
        #print("label", label)

        encoded_input = self.tokenizer(
                text,
                max_length = self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids":input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        label_input ={
            "sarcasm": torch.tensor(label, dtype=torch.float),
        }

        return data_input, label_input

    def __len__(self):
        return self.df.shape[0]


class TestDataset(Dataset):
    def __init__(self, df, pretraine_path='xlm-roberta-base', max_length=128):
        self.df = df
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    def __getitem__(self, index):
        text = self.df.iloc[index]["tweet"]

        encoded_input = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids": input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        return data_input

    def __len__(self):
        return self.df.shape[0]


class MultiLTTrainDataset(Dataset):
    def __init__(self, df, pretraine_path='xlm-roberta-base', max_length=128):
        self.df = df
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    def __getitem__(self, index):
        text = self.df.iloc[index]['tweet']
        categories = self.df.iloc[index][['sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question']].values.tolist()
        #print("label", label)

        encoded_input = self.tokenizer(
                text,
                max_length = self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids":input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        label_input ={
            "sarcasm": torch.tensor(categories, dtype=torch.float)
        }

        return data_input, label_input

    def __len__(self):
        return self.df.shape[0]
