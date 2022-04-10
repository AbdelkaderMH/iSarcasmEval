import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import Dataset
import os
import numpy as np
from emoji import UNICODE_EMOJI
import TweetNormalizer
import re
import text_normalization


dic = {
      "egypt": 'المصرية',
	  "nile": 'المصرية',
	  "msa": "اللغة العربية الفصحى",
	  "magreb": "المغربية",
	  "gulf": "الخليجية",
	  "levant": "الشامية"
}

def is_emoji(s):
    return s in UNICODE_EMOJI

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

def preprocess(text, lang='ar'):
    if lang == 'ar':
        sent = add_space(text)
        sent = re.sub(r'(?:@[\w_]+)', "user", sent)
        sent = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "url", sent)
        sent = sent.replace('_', ' ')
        sent = sent.replace('#', ' ')
    else:
        sent = add_space(text)
        sent = re.sub(r'(?:@[\w_]+)', "@user", sent)
        sent = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "http", sent)
        sent = sent.replace('_', ' ')
        sent = sent.replace('#', ' ')

    return sent

def prepare_text(df, col='tweet'):
    if col == 'tweet':
        df['dialect'] = df['dialect'].map(dic)   
    for i in range(df.shape[0]):
        df.loc[i, col] = df.loc[i, 'dialect'] + ' [SEP] ' + df.loc[i, col]


    return df

def augment_data(df_train):
    df_aug = pd.DataFrame(columns=['tweet', 'sarcastic'])
    dic_dup = {1: 3,
               0: 1
               }
    for i in range(df_train.shape[0]):
        current = df_train.iloc[i]
        text = current['tweet']
        label_cat = current['sarcastic']

        aug_ratio = dic_dup[label_cat]
        for k in range(aug_ratio):
            tokens = text.split(' ')
            l = len(tokens)
            n = int(0.1 * l)
            indices = np.random.choice(l, n, replace=False)
            for j in range(len(indices)):
                tokens[indices[j]] = '[MASK]'
            new_text = ' '.join(tokens)
            entry = {'tweet': new_text, 'sarcastic': label_cat}
            df_aug = df_aug.append(entry, ignore_index=True)
    df_aug.drop_duplicates(subset=['tweet'], keep='first', inplace=True)
    df = pd.concat([df_train,df_aug])
    return df

def loadTrainValData(lang='ar', size=0.2, batchsize=16, num_worker=0, pretraine_path="xlm-roberta-base", seed=42):

    path = "data/train.Ar.csv" if lang =='ar' else "data/train.En.csv"

    data = pd.read_csv(path, encoding='utf-8')
    data = data[~data.tweet.isna()]
    data['tweet'] = data['tweet'].apply(lambda x: preprocess(x, lang))
    if lang =='ar':
        data = prepare_text(data)
    data = data[["tweet", "sarcastic"]]
    data = data.sample(frac=1).reset_index(drop=True)
    df_train, df_test = train_test_split(data, test_size=size, stratify=data["sarcastic"].values, random_state=42, shuffle=True)#
    #df_train.dropna(axis=0, inplace=True)
    #df_test.dropna(axis=0, inplace=True)

    print("Training labels distribution\n", df_train["sarcastic"].value_counts())
    print("Test labels distribution\n", df_test["sarcastic"].value_counts())


    DF_train = Dataset.TrainDataset(df_train, pretraine_path)
    DF_test = Dataset.TrainDataset(df_test, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_train_loader, DF_test_loader



def loadMultiTrainValData(size=0.1, batchsize=16, num_worker=0, pretraine_path="xlm-roberta-base", seed=42):

    path = "data/train.En.csv"

    data = pd.read_csv(path, encoding='utf-8')
    data = data[["tweet", "sarcastic", 'sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question']]
    data = data[~data.tweet.isna()]

    data = data.dropna()

    data = data.sample(frac=1).reset_index(drop=True)
    df_train, df_test = train_test_split(data, test_size=size, random_state=42, shuffle=True)#


    df_train['tweet'] = df_train['tweet'].apply(lambda x: preprocess(x, lang='en'))

    df_test['tweet'] = df_test['tweet'].apply(lambda x: preprocess(x, lang='en'))



    DF_train = Dataset.MultiLTTrainDataset(df_train, pretraine_path)
    DF_test = Dataset.MultiLTTrainDataset(df_test, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_train_loader, DF_test_loader





def loadTrainValData2(lang='ar', size=0.2, batchsize=16, num_worker=0, pretraine_path="xlm-roberta-base", seed=42):

    path = "data/train.Ar.csv" if lang =='ar' else "data/train.En.csv"

    data = pd.read_csv(path, encoding='utf-8')
    data = data[~data.tweet.isna()]
    data['tweet'] = data['tweet'].apply(lambda x: preprocess(x, lang=lang))
    if lang =='ar':
        data = prepare_text(data, col='tweet')
        rephrase_df = data[["rephrase", "dialect"]]
        rephrase_df = rephrase_df.dropna().reset_index(drop=True)
        rephrase_df['rephrase'] = rephrase_df['rephrase'].apply(lambda x: preprocess(x, lang=lang))
        rephrase_df = prepare_text(rephrase_df, col='rephrase')
        rephrase_df = rephrase_df[["rephrase"]]
        print(rephrase_df.shape)
    else:    
         rephrase_df = data[["rephrase"]]
         rephrase_df = rephrase_df.dropna()
         rephrase_df['rephrase'] = rephrase_df['rephrase'].apply(lambda x: preprocess(x, lang=lang))
         print(rephrase_df.shape)
    rephrase_df["sarcastic"] = 0
    rephrase_df.columns = ["tweet", "sarcastic"]
    data = data[["tweet", "sarcastic"]]
    data = data.sample(frac=1).reset_index(drop=True)
    df_train, df_test = train_test_split(data, test_size=size, stratify=data["sarcastic"].values, random_state=42, shuffle=True)#
    print("Training labels distribution\n", df_train["sarcastic"].value_counts())
    print("Test labels distribution\n", df_test["sarcastic"].value_counts())
    df_train = pd.concat([df_train, rephrase_df])
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    print("Training labels distribution\n", df_train["sarcastic"].value_counts())
    print("Test labels distribution\n", df_test["sarcastic"].value_counts())


    DF_train = Dataset.TrainDataset(df_train, pretraine_path)
    DF_test = Dataset.TrainDataset(df_test, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_train_loader, DF_test_loader



def loadALLTrainValData(lang, size=0.1, batchsize=16, num_worker=0, pretraine_path="xlm-roberta-base", seed=42):

    path1 = "data/train.Ar.csv"
    path2= "data/train.En.csv"

    data1 = pd.read_csv(path1, encoding='utf-8')
    data1 = data1[["tweet", "sarcastic"]]
    data1 = data1[~data1.tweet.isna()]
    data1 = data1.sample(frac=1).reset_index(drop=True)
    df_train1, df_test1 = train_test_split(data1, test_size=size, stratify=data1["sarcastic"].values, random_state=42, shuffle=True)#

    data2 = pd.read_csv(path2, encoding='utf-8')
    data2 = data2[["tweet", "sarcastic"]]
    data2 = data2[~data2.tweet.isna()]
    data2 = data2.sample(frac=1).reset_index(drop=True)
    df_train2, df_test2 = train_test_split(data2, test_size=size, stratify=data2["sarcastic"].values, random_state=42, shuffle=True)#

    #df_train.dropna(axis=0, inplace=True)
    #df_test.dropna(axis=0, inplace=True)
    df_test = pd.concat([df_test1, df_test2])
    df_train = pd.concat([df_train1, df_train2])

    df_train['tweet'] = df_train['tweet'].apply(lambda x: preprocess(x,lang=lang))
    df_test['tweet'] = df_test['tweet'].apply(lambda x: preprocess(x, lang=lang))
    print("Training labels distribution\n", df_train["sarcastic"].value_counts())
    print("Test labels distribution\n", df_test["sarcastic"].value_counts())


    DF_train = Dataset.TrainDataset(df_train, pretraine_path)
    DF_test = Dataset.TrainDataset(df_test, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_train_loader, DF_test_loader



def loadTestData(lang='ar', batchsize=16, num_worker=2, pretraine_path="xlm-roberta-base"):
    path = "data/task_A_AR_test.csv" if lang == 'ar' else "data/taskA.En.input.csv"
    data = pd.read_csv(path, encoding='utf-8')
    data['tweet'] = data['tweet'].apply(lambda x:preprocess(x, lang=lang))
    print(data.shape)
    print(data.head())
    if lang =='ar':
        data = prepare_text(data, col='tweet')

    DF_test = Dataset.TestDataset(data, pretraine_path)

    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_test_loader

def loadTestData2(column, lang='ar', batchsize=16, num_worker=2, pretraine_path="xlm-roberta-base"):
    path = "data/task_C_AR_test.csv" if lang == 'ar' else "data/taskC.En.input.csv"
    data = pd.read_csv(path, encoding='utf-8')
    print(data.shape)
    data['tweet'] = data[column].apply(lambda x:preprocess(x, lang=lang))
    if lang =='ar':
        data = prepare_text(data, col='tweet')


    DF_test = Dataset.TestDataset(data, pretraine_path)

    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_test_loader


def loadMultiTestData(batchsize=16, num_worker=2, pretraine_path="xlm-roberta-base"):
    path = "data/taskB.En.input.csv"
    data = pd.read_csv(path, encoding='utf-8')
    data['tweet'] = data['tweet'].apply(lambda x:preprocess(x, lang='en'))

    DF_test = Dataset.TestDataset(data, pretraine_path)

    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_test_loader

