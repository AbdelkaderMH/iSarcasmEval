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
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report
from train_cgan import predict
import utils
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)
def train(base_model, mt_classifier, generator, iterator, optimizer, optimizer_g, sar_criterion, scheduler, scheduler_g):

    # set the model in eval phase
    base_model.train(True)
    mt_classifier.train(True)
    generator.train(True)
    acc_sarcasm= 0
    loss_sarc= 0
    loss_disc = 0
    loss_gen = 0

    for data_input, label_input  in Bar(iterator):

        for k, v in data_input.items():
            data_input[k] = v.to(device)

        for k, v in label_input.items():
            label_input[k] = v.to(device)

        optimizer_g.zero_grad()

        sarcasm_target = label_input['sarcasm']
        batch_size = sarcasm_target.shape[0]
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, 100)))).to(device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 20, batch_size))).to(device)
        fake_features = generator(z, fake_labels)
        fake_logits = mt_classifier(fake_features)
        fake_logits = fake_logits[:, -1]

        g_target = torch.from_numpy(np.array([1] * batch_size)).float().to(device)
        r_target = torch.from_numpy(np.array([0] * batch_size)).float().to(device)
        gen_loss = F.binary_cross_entropy_with_logits(fake_logits, r_target)
        loss_gen += gen_loss.item()
        gen_loss.backward()
        optimizer_g.step()
        scheduler_g.step()

        ## dicriminator/classifier backward
        optimizer.zero_grad()
        # compute the loss
        output = base_model(**data_input)
        sarcasm_logits = mt_classifier(output)
        fake_logits = mt_classifier(fake_features.detach())
        fake_logits = fake_logits[:, -1]
        sarcasm_probs = torch.sigmoid(sarcasm_logits[:, :6])
        #_, s_tgt, _ = torch.svd(sarcasm_probs)
        #transfer_loss = -torch.mean(s_tgt)


        real_logits = sarcasm_logits[:, -1]

        loss_sarcasm = sar_criterion(sarcasm_logits[:,:6], sarcasm_target)
        dics_loss =  0.5 * (F.binary_cross_entropy_with_logits(real_logits, r_target) +
                           F.binary_cross_entropy_with_logits(fake_logits, g_target))

        total_loss = loss_sarcasm + dics_loss
        #total_loss = multi_task_loss(loss_sentiment, loss_sarcasm)
        loss_sarc += loss_sarcasm.item()
        loss_disc += dics_loss.item()
        # backpropage the loss and compute the gradients
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        acc_sarcasm += utils.binary_accuracy(sarcasm_probs, sarcasm_target)

    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator)}
    losses = { 'Sarcasm': loss_sarc / len(iterator), 'Discriminator': loss_disc / len(iterator), 'Generator': loss_gen / len(iterator)}
    return accuracies, losses

def evaluate(base_model, mt_classifier, generator,iterator, sar_criterion):
    # initialize every epoch
    acc_sarcasm= 0
    loss_sarc= 0
    loss_disc = 0
    loss_gen = 0

    #all_sarcasm_outputs = []
    all_sarcasm_outputs = np.empty(shape=[0, 6])
    all_sarcasm_labels = np.empty(shape=[0, 6])

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    generator.eval()
    with torch.no_grad():
        for data_input, label_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            for k, v in label_input.items():
                label_input[k] = v.to(device)


            sarcasm_target = label_input['sarcasm']
            #print("shape sarcasm labels", sarcasm_target.shape[0])
            # forward pass
            batch_size = sarcasm_target.shape[0]
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, 100)))).to(device)
            fake_labels = Variable(torch.LongTensor(np.random.randint(0, 20, batch_size))).to(device)
            fake_features = generator(z, fake_labels)
            fake_logits = mt_classifier(fake_features)
            f_logits = fake_logits[:, -1]
            g_target = torch.from_numpy(np.array([1] * batch_size)).float().to(device)
            r_target = torch.from_numpy(np.array([0] * batch_size)).float().to(device)
            #print('f_logits', f_logits.shape)
            #print('r_target', r_target.shape)
            gen_loss = F.binary_cross_entropy_with_logits(f_logits, r_target)
            loss_gen += gen_loss.item()



            output = base_model(**data_input)
            sarcasm_logits = mt_classifier(output)
            sarcasm_probs = torch.sigmoid(sarcasm_logits[:, :6])
            real_logits = sarcasm_logits[:, -1]

            loss_sarcasm = sar_criterion(sarcasm_logits[:, :6], sarcasm_target)
            dics_loss = 0.5 * (F.binary_cross_entropy_with_logits(real_logits, r_target) +
                               F.binary_cross_entropy_with_logits(f_logits, g_target))

            total_loss = loss_sarcasm + dics_loss
            # total_loss = multi_task_loss(loss_sentiment, loss_sarcasm)
            loss_sarc += loss_sarcasm.item()
            loss_disc += dics_loss.item()
            #mtl_loss = multi_task_loss(loss_sentiment, loss_sarcasm)

            # compute the running accuracy and losses
            acc_sarcasm += utils.binary_accuracy(sarcasm_probs, sarcasm_target)


            loss_sarc += loss_sarcasm.item()

            predicted_sarcasm = torch.round(sarcasm_probs)
            #all_sarcasm_outputs.extend(predicted_sarcasm.squeeze().int().cpu().numpy().tolist())
            all_sarcasm_outputs = np.append(all_sarcasm_outputs, predicted_sarcasm.squeeze().cpu().numpy(), axis=0)
            #all_sarcasm_labels.extend(sarcasm_target.squeeze().int().cpu().numpy().tolist())
            all_sarcasm_labels = np.append(all_sarcasm_labels, sarcasm_target.squeeze().cpu().numpy(), axis=0)
    all_sarcasm_outputs = all_sarcasm_outputs.reshape(-1, 6)
    all_sarcasm_labels = all_sarcasm_labels.reshape(-1, 6)
    report_sarcasm = classification_report(y_true=all_sarcasm_labels, y_pred=all_sarcasm_outputs,digits=4)
    fscore = f1_score(y_true=all_sarcasm_labels, y_pred=all_sarcasm_outputs, average='macro')


    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator),  'Report_sarcasm': report_sarcasm, 'F1-Sarcasm': fscore}
    losses = { 'Total': (loss_sarc + loss_disc) / len(iterator) ,'Sarcasm': loss_sarc / len(iterator), 'Discriminator': loss_disc / len(iterator), 'Generator': loss_gen / len(iterator)}
    return accuracies, losses


def predict(base_model, mt_classifier,iterator):
    # initialize every epoch

    all_sarcasm_outputs = np.empty(shape=[0, 6])

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)


            output = base_model(**data_input)
            sarcasm_logits = mt_classifier(output)
            sarcasm_probs = torch.sigmoid(sarcasm_logits[:, :6])


            # compute the running accuracy and losses
            predicted_sarcasm = torch.round(sarcasm_probs)
            all_sarcasm_outputs = np.append(all_sarcasm_outputs, predicted_sarcasm.squeeze().cpu().numpy(), axis=0)

    all_sarcasm_outputs = all_sarcasm_outputs.reshape(-1, 6)

    return all_sarcasm_outputs

def eval_full(config, loader):
    criterion = config['loss']
    base_model = modeling.TransformerLayerG(pretrained_path=config['pretrained_path'], both=True).to(device)
    classifier = modeling.Classifier(in_feature=base_model.output_num(), hidden_size= 512, class_num=7).to(device)
    base_model.load_state_dict(torch.load(f"./ckpts/best_basemodel_sarcasm__{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_multi_cgan_en.pth"))
    classifier.load_state_dict(torch.load(f"./ckpts/best_cls_sarcasm__{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_multi_cgan_en.pth"))
    base_model = base_model.to(device)
    classifier = classifier.to(device)
     
    all_outputs = predict(base_model, classifier, loader)
    df = pd.DataFrame(all_outputs, columns=['sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question']) 
    df = df.astype(int)
    df.to_csv(f'results/task_b_en_{criterion}_cgan.txt', index=False, header=True)


def train_full(config, train_loader, stest_loader):
    lr_o = config['lr_mult'] * config['lr']
    lr = config['lr']
    criterion= config['loss']
    #Instanciate models
    base_model = modeling.TransformerLayerG(pretrained_path=config['pretrained_path'], both=True).to(device)
    mtl_classifier = modeling.Classifier(in_feature=base_model.output_num(), hidden_size= 512, class_num=7).to(device)
    generator = modeling.Generator(in_feature=164, hidden_size=base_model.output_num(), num_class=20, emsize=64).to(device)
    cls = 'ATTClassifier'

    if criterion =='WBCE':
        sarc_criterion = focal_loss.WeightedBCELoss().to(device)
    elif criterion == 'FL':
        sarc_criterion = focal_loss.BinaryFocalLoss(alpha=0.1, gamma=2).to(device)
    else:
        sarc_criterion = nn.BCEWithLogitsLoss().to(device)


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_base_cls = [{"params": base_model.parameters()}, {'params': mtl_classifier.parameters(), 'lr':  config['lr']}]

    optimizer = AdamW(param_base_cls, lr=config["lr"])
    optimizer_g = AdamW(generator.parameters(),  lr= 3 * config['lr'])
    train_data_size = len(train_loader)
    steps_per_epoch = int(train_data_size / config['batch_size'])
    num_train_steps = train_data_size * config['epochs']
    warmup_steps = num_train_steps * 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    scheduler_g = get_linear_schedule_with_warmup(optimizer_g,  num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    # Train model
    best_sentiment_valid_accuracy, best_sarcasm_valid_accuracy = 0, 0
    best_total_val_acc = 0
    best_val_loss = float('+inf')
    report_sarcasm = None

    epo = 0
    for epoch in range(config['epochs']):
        print("epoch {}".format(epoch + 1))

        train_accuracies, train_losses = train(base_model, mtl_classifier, generator, train_loader, optimizer, optimizer_g, sarc_criterion,scheduler, scheduler_g)
        valid_accuracies, valid_losses = evaluate(base_model, mtl_classifier, generator, valid_loader, sarc_criterion)
        #print(multi_task_loss.parameters())
        val_loss = valid_losses['Total']
        total_val_acc = valid_accuracies['F1-Sarcasm']
        if total_val_acc > best_total_val_acc:
        #if best_val_loss > val_loss:
            epo = epoch
            best_val_loss = val_loss
            report_sarcasm = valid_accuracies['Report_sarcasm']
            best_sarcasm_loss = valid_losses['Total']
            best_total_val_acc = total_val_acc
            best_sarcasm_valid_accuracy = total_val_acc
            print("save model's checkpoint")
            torch.save(base_model.state_dict(), f"./ckpts/best_basemodel_sarcasm__{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_multi_cgan_en.pth")
            torch.save(mtl_classifier.state_dict(), f"./ckpts/best_cls_sarcasm__{config['args'].lm_pretrained}_{config['args'].lang}_{criterion}_multi_cgan_en.pth")


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
        for k, v in valid_accuracies.items():
            if 'Report' not in k:
                print(k+f": {v * 100:.2f}")
        print("losses**********")
        for k, v in valid_losses.items():
            print(k + f": {v :.5f}\t")
        print('******************************************************\n')
    print(f"epoch of best results {epo}")
    with open(f'reports/report_Sarcasm_{config["args"].lm_pretrained}_{config["args"].lang}_cat_{criterion}_cgan_multi.txt', 'w') as f:
        f.write("Sarcasm report\n")
        f.write(report_sarcasm)
        f.write(str(config['args']))
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
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--loss', type=str, default="WBCE", choices=['WBCE', 'FL', 'BCE'])
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
    if args.lm_pretrained == 'roberta':
        config['pretrained_path'] = "cardiffnlp/twitter-roberta-base"
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
            train_loader, valid_loader = preprocessing.loadMultiTrainValData(batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        elif args.lang == 'ar':
            print("error english only")
            exit()

        best_sarcasm_acc, best_sarcasm_loss =train_full(config, train_loader, valid_loader)
        print(f'  Val. Sarcasm F1: {best_sarcasm_acc * 100:.2f}%  \t Val Sarcasm Loss {best_sarcasm_loss :.4f} ')
    else:
        test_loader = preprocessing.loadMultiTestData(batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        eval_full(config, test_loader)