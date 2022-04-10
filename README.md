Eval-2022 Task 6: Transformer-based Models for Intended Sarcasm Detection in English and Arabic

This repository releases the PyTorch implementation of our submitted system to the intended sarcasm detection task in English and Arabic languages (Task 6: iSarcasmEval of SemEval-2022).


### How to use the code:
#### 0. Prerequisites

We carry out experiments using the following packages.

```
Python 3
sklearn
PyTorch
transformers
emoji
```

#### 1. Prepare datasets

put the shared task csv files into ./data folder

#### 2. Model training/inference

###### Task A:
Hyperparameters list:
lm : 'marbert'  for AR or 'twitter' for EN (cardiffnlp/twitter-xlm-roberta-base)
phase: train for model training (predict for inference), predict for model prediction, and taskc for Task C prediction
lang : ar or en 
loss: - WBCE, FL, BCE (Model 1)
      - WCE, FL, CE (Model 2, and 3)
```
####### Model 1
python train_sar.py --lm_pretrained ${lm} --lr ${lr} --batch_size ${size} --epochs ${epochs}  --phase ${phase} --lang ${lang} --loss ${loss}

####### Model 2
python train_sarcat.py --lm_pretrained ${lm} --lr ${lr} --batch_size ${size} --epochs ${epochs}  --phase ${phase} --lang ${lang} --loss ${loss}

####### Model 3
python train_cgan.py --lm_pretrained ${lm} --lr ${lr} --batch_size ${size} --epochs ${epochs}  --phase ${phase} --lang ${lang} --loss ${loss}

```

###### Task B:

loss: WBCE, FL, BCE
```
python train_cgan_multi.py --lm_pretrained ${lm} --lr ${lr} --batch_size ${size} --epochs ${epochs}  --phase ${phase} --lang ${lang}s

```




Please cite the following paper if you use this code in your research.

      @inproceedings{el-mahdaouy-etal-2022-cs,
      Author = {Abdelkader {El Mahdaouy},  Abdellah {El Mekki} , Kabil Essefar, Abderrahman Skiredj, and Ismail Berrada},
      Title = {CS-UM6P at SemEval-2022 Task 6: Transformer-based Models for Intended Sarcasm Detection in English and Arabic},
      Year = {2022},
      booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
      }
