import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import preprocessing
import modeling2 as modeling
from barbar import Bar
import random
import focal_loss

from sklearn.metrics import f1_score, accuracy_score, classification_report
import utils
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(base_model, mt_classifier, iterator, optimizer, sar_criterion, scheduler):

    # set the model in eval phase
    base_model.train(True)
    mt_classifier.train(True)
    acc_sarcasm= 0
    loss_sarc= 0

    for data_input, label_input  in Bar(iterator):

        for k, v in data_input.items():
            data_input[k] = v.to(device)

        for k, v in label_input.items():
            label_input[k] = v.long().to(device)

        optimizer.zero_grad()


        #forward pass

        sarcasm_target = label_input['sarcasm']

        # forward pass

        output = base_model(**data_input)
        sarcasm_logits = mt_classifier(output)

        sarcasm_probs = torch.softmax(sarcasm_logits, dim=1)
        # compute the loss
        #print("logit device", sarcasm_logits.get_device())
        #print("labels device", sarcasm_target.get_device())
        loss_sarcasm = sar_criterion(sarcasm_logits, sarcasm_target)
        #total_loss = multi_task_loss(loss_sentiment, loss_sarcasm)
        loss_sarc += loss_sarcasm.item()
        # backpropage the loss and compute the gradients
        loss_sarcasm.backward()
        optimizer.step()
        scheduler.step()
        acc_sarcasm += utils.calc_accuracy(sarcasm_probs, sarcasm_target)

    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator)}
    losses = { 'Sarcasm': loss_sarc / len(iterator)}
    return accuracies, losses

def evaluate(base_model, mt_classifier, iterator, sar_criterion):
    # initialize every epoch
    acc_sarcasm= 0
    loss_sarc= 0

    #all_sarcasm_outputs = []
    all_sarcasm_outputs = np.array([])
    all_sarcasm_labels = np.array([])

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    with torch.no_grad():
        for data_input, label_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            for k, v in label_input.items():
                label_input[k] = v.long().to(device)


            sarcasm_target = label_input['sarcasm']

            # forward pass

            output = base_model(**data_input)
            sarcasm_logits = mt_classifier(output)
            logits = sarcasm_logits[:,:2]
            #print("final logits shape ", logits.shape)

            sarcasm_probs = torch.softmax(logits, dim=1)
            # compute the loss
            #print("logit device", sarcasm_logits.get_device())
            #print("labels device", sarcasm_target.get_device())
            loss_sarcasm = sar_criterion(logits, sarcasm_target)
            #mtl_loss = multi_task_loss(loss_sentiment, loss_sarcasm)

            # compute the running accuracy and losses
            acc_sarcasm += utils.calc_accuracy(sarcasm_probs, sarcasm_target)


            loss_sarc += loss_sarcasm.item()

            _, predicted_sarcasm = torch.max(sarcasm_probs, 1)
            #all_sarcasm_outputs.extend(predicted_sarcasm.squeeze().int().cpu().numpy().tolist())
            all_sarcasm_outputs = np.append(all_sarcasm_outputs, predicted_sarcasm.squeeze().cpu().numpy())
            #all_sarcasm_labels.extend(sarcasm_target.squeeze().int().cpu().numpy().tolist())
            all_sarcasm_labels = np.append(all_sarcasm_labels, sarcasm_target.squeeze().cpu().numpy())
    all_sarcasm_outputs = all_sarcasm_outputs.reshape(-1)
    all_sarcasm_labels = all_sarcasm_labels.reshape(-1)
    fscore_sarcasm = f1_score(y_true=all_sarcasm_labels, y_pred=all_sarcasm_outputs, average='macro')
    report_sarcasm = classification_report(y_true=all_sarcasm_labels, y_pred=all_sarcasm_outputs,digits=4)


    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator), 'F1_sarcasm': fscore_sarcasm, 'Report_sarcasm': report_sarcasm}
    losses = { 'Sarcasm': loss_sarc / len(iterator)}
    return accuracies, losses


def predict(base_model, mt_classifier, iterator):
    # initialize every epoch
    acc_sarcasm= 0
    loss_sarc= 0

    #all_sarcasm_outputs = []
    all_sarcasm_outputs = np.array([])
    all_sarcasm_labels = np.array([])

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            # forward pass

            output = base_model(**data_input)
            sarcasm_logits = mt_classifier(output)
            logits = sarcasm_logits[:,:2]
            #print("final logits shape ", logits.shape)

            sarcasm_probs = torch.softmax(logits, dim=1)
            # compute the loss
            #print("logit device", sarcasm_logits.get_device())
            #print("labels device", sarcasm_target.get_device())

            _, predicted_sarcasm = torch.max(sarcasm_probs, 1)
            #all_sarcasm_outputs.extend(predicted_sarcasm.squeeze().int().cpu().numpy().tolist())
            all_sarcasm_outputs = np.append(all_sarcasm_outputs, predicted_sarcasm.squeeze().cpu().numpy())

    all_sarcasm_outputs = all_sarcasm_outputs.reshape(-1)

    return all_sarcasm_outputs

def score(base_model, mt_classifier, iterator):
    # initialize every epoch


    all_sarcasm_outputs = np.array([])

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            # forward pass

            output = base_model(**data_input)
            sarcasm_logits = mt_classifier(output)
            logits = sarcasm_logits[:,:2]
            sarcasm_probs = torch.softmax(logits, dim=1)
            predicted_sarcasm = sarcasm_probs[:,1]

            all_sarcasm_outputs = np.append(all_sarcasm_outputs, predicted_sarcasm.squeeze().cpu().numpy())

    all_sarcasm_outputs = all_sarcasm_outputs.reshape(-1)

    return all_sarcasm_outputs

def eval_full(config, loader1, loader2=None, score=False):
    criterion = config['loss']
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    classifier = modeling.ATTClassifier(base_model.output_num(), class_num=2).to(device)
    base_model.load_state_dict(torch.load(f"./ckpts/best_basemodel_sarcasm_{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_sarcat.pth"))
    classifier.load_state_dict(torch.load(f"./ckpts/best_cls_sarcasm_{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_sarcat.pth"))
    base_model = base_model.to(device)
    classifier = classifier.to(device)
    df = pd.DataFrame()  
    if not score:
        col = 'task_a_ar' if config['args'].lang =='ar' else 'task_a_en'
        all_outputs = predict(base_model, classifier, loader1)
        df[col] = all_outputs
        df = df.astype(int)
        df.to_csv(f'results/{col}_{criterion}_sarcat.txt', index=False, header=True)

    else:
        col = 'task_c_ar' if config['args'].lang =='ar' else 'task_c_en'
        all_outputs1 = predict(base_model, classifier, loader1)
        all_outputs2 = predict(base_model, classifier, loader2)
        all_outputs = []
        for i in range(len(all_outputs1)):
            if all_outputs1[i] >= all_outputs2[i]:
                all_outputs.append(0)
            else:
                 all_outputs.append(1)
        df[col] = all_outputs
        df.to_csv(f'results/{col}_{criterion}_sarcat.txt', index=False, header=True)




def train_full(config, train_loader, stest_loader):
    lr_o = config['lr_mult'] * config['lr']
    lr = config['lr']
    criterion = config['loss']
    #Instanciate models
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    mtl_classifier = modeling.ATTClassifier(base_model.output_num(), class_num=2).to(device)
    cls = 'ATTClassifier'


    if criterion == 'WCE':
        sarc_criterion = focal_loss.WeightedCELoss().to(device)
    elif criterion =='FL':
        sarc_criterion = focal_loss.FocalLoss_Ori(num_class=2, alpha=[1, 1], gamma=2).to(device)
    else:
        sarc_criterion = nn.CrossEntropyLoss().to(device)
    #
    
    params = [{'params':base_model.parameters(), 'lr':config['lr']}, {'params': mtl_classifier.parameters(), 'lr': config['lr']}]#, {'params':multi_task_loss.parameters(), 'lr': 0.0005}]
    optimizer = AdamW(params, lr=config["lr"])
    train_data_size = len(train_loader)
    steps_per_epoch = int(train_data_size / config['batch_size'])
    num_train_steps = len(train_loader) * config['epochs']
    warmup_steps = int(config['epochs'] * train_data_size * 0.1 / config['batch_size'])
    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    # Train model
    best_sentiment_valid_accuracy, best_sarcasm_valid_accuracy = 0, 0
    best_total_val_acc = 0
    best_val_loss = float('+inf')
    report_sarcasm = None
    epo = 0
    for epoch in range(config['epochs']):
        print("epoch {}".format(epoch + 1))

        train_accuracies, train_losses = train(base_model, mtl_classifier, train_loader, optimizer, sarc_criterion,scheduler)
        valid_accuracies, valid_losses = evaluate(base_model, mtl_classifier, valid_loader, sarc_criterion)
        #print(multi_task_loss.parameters())
        val_loss = valid_losses['Sarcasm']
        total_val_acc = valid_accuracies['F1_sarcasm']
        if total_val_acc > best_total_val_acc:
        #if best_val_loss > val_loss:
            epo = epoch
            best_val_loss = val_loss
            best_total_val_acc = total_val_acc
            best_sarcasm_valid_accuracy = valid_accuracies['F1_sarcasm']
            report_sarcasm = valid_accuracies['Report_sarcasm']
            best_sarcasm_loss = valid_losses['Sarcasm']
            print("save model's checkpoint")
            torch.save(base_model.state_dict(), f"./ckpts/best_basemodel_sarcasm_{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_sarcat.pth")
            torch.save(mtl_classifier.state_dict(), f"./ckpts/best_cls_sarcasm_{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_sarcat.pth")


        print('********************Train Epoch***********************\n')
        print("accuracies**********")
        for k , v in train_accuracies.items():
            print(k+f" : {v * 100:.2f}")
        print("losses**********")
        for k , v in train_losses.items():
            print(k+f": {v :.5f}\t")
        print('********************Validation***********************\n')
        print("accuracies**********")
        print(valid_accuracies['Report_sarcasm'])
        #for k, v in valid_accuracies.items():
        #    if 'Report' not in k:
        #        print(k+f": {v * 100:.2f}")
        print("losses**********")
        for k, v in valid_losses.items():
            print(k + f": {v :.5f}\t")
        print('******************************************************\n')
    print(f"epoch of best results {epo}")
    with open(f'reports/report_Sarcasm_{config["args"].lm_pretrained}_{config["args"].lang}_cat_{criterion}.txt', 'w') as f:
        f.write("Sarcasm report\n")
        f.write(report_sarcasm)
    return best_sarcasm_valid_accuracy, best_sarcasm_loss
if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--lm_pretrained', type=str, default='arabert',
                        help=" path of pretrained transformer")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_mult', type=float, default=1, help="dicriminator learning rate multiplier")

    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--lang', type=str, default="ar")
    parser.add_argument('--loss', type=str, default="WCE", choices=['WCE', 'FL', 'CE'])
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()


    config = {}
    config['args'] = args
    config["output_for_test"] = True
    config['epochs'] = args.epochs
    config["class_num"] = 1
    config["lr"] = args.lr
    config['lr_mult'] = args.lr_mult
    config['batch_size'] = args.batch_size
    config['lm'] = args.lm_pretrained
    config['loss'] = args.loss
    lang = args.lang
    dosegmentation = False
    if args.lm_pretrained == 'xlml':
        config['pretrained_path'] = "xlm-roberta-large"
    elif args.lm_pretrained == 'marbert':
        config['pretrained_path'] = "UBC-NLP/MARBERT"
    elif args.lm_pretrained == 'xlm':
        config['pretrained_path'] = "xlm-roberta-base"
        dosegmentation = True
    elif args.lm_pretrained == 'twitter':
        config['pretrained_path'] = "cardiffnlp/twitter-xlm-roberta-base"
        dosegmentation = True
    else:
        config['pretrained_path'] = 'lanwuwei/GigaBERT-v3-Arabic-and-English'
        dosegmentation = True

    RANDOM_SEED = 3407#, 12346, 12347, 12348, 12349]

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.phase == 'train':
        if args.lang == 'en':
            train_loader, valid_loader = preprocessing.loadTrainValData2(lang=lang, batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        elif args.lang == 'ar':
            train_loader, valid_loader = preprocessing.loadTrainValData2(lang=lang, batchsize=args.batch_size,
                                                                         num_worker=0, pretraine_path=config['pretrained_path'])
        else:
            train_loader, valid_loader = preprocessing.loadALLTrainValData( batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])

        best_sarcasm_acc, best_sarcasm_loss =train_full(config, train_loader, valid_loader)
        print(f'  Val. Sarcasm F1: {best_sarcasm_acc * 100:.2f}%  \t Val Sarcasm Loss {best_sarcasm_loss :.4f} ')
    elif args.phase == 'predict':
        test_loader = preprocessing.loadTestData(lang= lang, batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        eval_full(config, loader1=test_loader, loader2=None, score=False)
    else:
        test_loader1 = preprocessing.loadTestData2(column='text_0',lang= lang, batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        test_loader2 = preprocessing.loadTestData2(column='text_1',lang= lang, batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        eval_full(config, loader1= test_loader1, loader2=test_loader2, score=True)