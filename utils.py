import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import accuracy_score, f1_score



def label_rule(mis_preds, cat_preds):
    for i in range(len(mis_preds)):
        #print(mis_preds[i], cat_preds[i])
        if mis_preds[i] == 0:
            cat_preds[i] = 0
    return cat_preds


def accuracy(preds, y):
    all_output = preds.float().cpu()
    all_label = y.float().cpu()
    _, predict = torch.max(all_output, 1)
    acc = accuracy_score(all_label.numpy(), torch.squeeze(predict).float().numpy())
    return acc

def calc_accuracy(preds,y):
    predict = torch.argmax(preds, dim=1)
    accuracy = torch.sum(predict == y.squeeze()).float().item()
    return accuracy / float(preds.size()[0])

def binary_accuracy2(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds).squeeze()

    correct = (rounded_preds == y).float()
    acc = correct.sum() / (y.size(0))
    return acc

def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds).squeeze()

    correct = (rounded_preds == y).float()
    acc = correct.sum() / (y.size(0) * y.size(1))
    return acc
