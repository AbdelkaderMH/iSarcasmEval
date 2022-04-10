import torch.nn as nn
import torch
from transformers import AutoModel
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)
        #self.apply(init_weights)
    def forward(self, inp):
        u = torch.tanh_(self.attn(inp))
        a = F.softmax(self.contx(u), dim=1)
        s = (a * inp).sum(1)
        return s


class TransformerLayer(nn.Module):
    def __init__(self,both=True,
                 pretrained_path='aubmindlab/bert-base-arabert'):
        super(TransformerLayer, self).__init__()

        self.both = both
        self.transformer = AutoModel.from_pretrained(pretrained_path, output_hidden_states=True)


    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #(output_last_layer, pooled_cls, (output_layers))
        #output[0] (8, seqlen=64, 768) cls [8, 768] ( 12 (8, seqlen=64, 768))

        return outputs

    def output_num(self):
        return self.transformer.config.hidden_size

class ATTClassifier(nn.Module):
    def __init__(self, in_feature, class_num=1, dropout_prob=0.2):
        super(ATTClassifier, self).__init__()
        self.attention = AttentionWithContext(in_feature)

        self.Classifier = nn.Sequential(
            nn.Linear(2 * in_feature, 512),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(512, class_num)
        )

        self.apply(init_weights)

    def forward(self, x):
        att = self.attention(x[0]) #(X[0] (bs, seqlenght, embedD) att = \sum_i alpha_i x[0][i]

        xx = torch.cat([att, x[1]], 1)

        out = self.Classifier(xx)
        return out



class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return - grad_output


def gradient_reversal_layer(x):
    return GradientReversalLayer.apply(x)


class Generator(nn.Module):
    def __init__(self, in_feature, hidden_size, num_class=2, emsize=120):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_class, emsize)
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.apply(init_weights)


    def forward(self, z, labels):
        c = self.label_emb(labels)
        #print('c shape', c.shape)
        #print("z shape", z.shape)
        x = torch.cat([z, c], 1)
        #print("x shape", x.shape)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        #x = gradient_reversal_layer(x)
        return x

    def output_num(self):
        return self.hidden_size



class TransformerLayerG(nn.Module):
    def __init__(self,both=True,
                 pretrained_path='aubmindlab/bert-base-arabert'):
        super(TransformerLayerG, self).__init__()
        self.path = pretrained_path
        self.both = both
        self.transformer = AutoModel.from_pretrained(pretrained_path, output_hidden_states=True)
        hidden_size = self.transformer.config.hidden_size
        self.attention = AttentionWithContext(hidden_size)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        att = self.attention(outputs[0])
        x = torch.cat([att, outputs[1]], 1)
        x = self.linear(x)

        return x

    def output_num(self):
        return self.transformer.config.hidden_size

class Classifier(nn.Module):
    def __init__(self, in_feature, hidden_size, class_num):
        super(Classifier, self).__init__()
        self.class_num = class_num
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.apply(init_weights)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return self.class_num


class ClassifierMulti(nn.Module):
    def __init__(self, in_feature, hidden_size, class_num, class_num2):
        super(ClassifierMulti, self).__init__()
        self.class_num = class_num
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, class_num)
        self.ad_layer4 = nn.Linear(hidden_size, class_num2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.apply(init_weights)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        l = self.ad_layer4(x)
        return y, l

    def output_num(self):
        return self.class_num

