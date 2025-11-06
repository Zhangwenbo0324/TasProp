import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch import Tensor

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, accuracy_score

import math, random, sys
import pickle
from optparse import OptionParser
import os.path

import pandas as pd
import numpy as np
from numpy import ndarray

sys.path.append(rf'/data-extend/liuxg/TasProp') # use your desired default path
sys.path.append(rf'/data-extend/liuxg/TasProp/jtnn') # use your desired default path
from jtnn import *
from jtnn import JTNNVAE, Vocab
import rdkit
from rdkit import Chem

sys.setrecursionlimit(10**5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
seed=42
setup_seed(seed)

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("--task", dest="task_name", default='bbbp')

parser.add_option("--origin_train_path", dest="origin_train_path")
parser.add_option("--gen_path", dest="gen_path")
parser.add_option("--test_path", dest="test_path")

parser.add_option("--vocab", dest="vocab_path")
parser.add_option("--model", dest="model_path", default=None)

parser.add_option("--batch", dest="batch_size", default=40)
parser.add_option("--hidden", dest="hidden_size", default=450)
parser.add_option("--latent", dest="latent_size", default=56)
parser.add_option("--depth", dest="depth", default=3)
parser.add_option("--alpha", dest="alpha", default=1.0)
parser.add_option("--beta", dest="beta", default=0.008)
parser.add_option("--lr", dest="lr", default=0.0007)
parser.add_option("--stereo", dest="stereo", default=1)
parser.add_option("--aug", dest="aug", default=1)

parser.add_option("--save_dir", dest="save_path")
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
alpha = float(opts.alpha)
lr = float(opts.lr)
stereo = True if int(opts.stereo) == 1 else False
aug = True if int(opts.aug) == 1 else False
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)
task_name = opts.task_name
save_path = opts.save_path

# IO function
def write_smiles(write_path, smiles:list, mode='w'):
    with open(write_path, mode=mode) as fp:
        for smile in smiles:
            if smile is None:
                continue
            fp.write(smile + "\n")

def write_label(write_path, labels:ndarray):
    with open (write_path, 'wb') as f:
        pickle.dump(labels, f)

def write_code(write_path, mol_code:Tensor):
    torch.save(mol_code, write_path)

def read_smiles(read_path) -> list:
    return [x.strip("\r\n ") for x in open(read_path)] 

def read_label(read_path) -> ndarray:
    with open (read_path, 'rb') as f: 
        labels = np.array(pickle.load(f)) 
    return labels

def read_code(read_path) -> Tensor:
    return torch.load(read_path)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pred = self.fc2(x)

        return torch.sigmoid(pred)

def get_CLSmodel(X_train, y_train, X_val, y_val, max_epoch=500, patience = 15):
    num, dim = X_train.size(0), X_train.size(1)
    best_val_auc = 0.0
    max_train_loss = 1e10
    best_model = None

    batchsize=40
    train_bs = int(math.ceil(num/batchsize))
    criterion = nn.BCELoss().to(device)
    out_dim = torch.max(y_train)
    
    model = MLP(dim,out_dim.item()+1).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    no_improvement_count = 0

    for epoch in range(max_epoch):
        model.train()

        for k in range(train_bs):  
            model.zero_grad()
            batch=X_train[k*batchsize:(k+1)*batchsize].to(device)
            target=y_train[k*batchsize:(k+1)*batchsize].to(device)
            pred = model.forward(batch.detach())
            if pred.squeeze().shape != target.view(-1).detach().shape:
                continue
            loss  = criterion(pred.squeeze(),target.view(-1).detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            val_bs =  int(math.ceil(X_val.size(0)/batchsize))

            all_preds = [] 
            all_labels = [] 

            for j in range(val_bs):  
                batch = X_val[j*batchsize:(j+1)*batchsize].to(device)
                target = y_val[j*batchsize:(j+1)*batchsize].to(device)
                pred = model.forward(batch.detach())
                if pred.squeeze().shape != target.view(-1).detach().shape:
                    continue
                all_preds.append(pred.cpu().numpy())
                all_labels.append(target.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            try:
                val_auc_roc = roc_auc_score(all_labels, all_preds)
            except:
                val_auc_roc = -1 

        if val_auc_roc > best_val_auc:
            best_val_auc = val_auc_roc
            best_model = model.state_dict().copy()
            best_e = epoch
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            break

    model.load_state_dict(best_model)
    model.eval()
    
    return model, best_val_auc

def test_model(model, X_test, y_test):
    batchsize = 40
    model.eval()
    test_bs =  int(math.ceil(X_test.size(0)/batchsize))

    all_preds = []  
    all_labels = []  
    with torch.no_grad():
        for j in range(test_bs):  
            batch = X_test[j*batchsize:(j+1)*batchsize].to(device)
            target = y_test[j*batchsize:(j+1)*batchsize].to(device)
            pred = model.forward(batch.detach())
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        threshold = 0.5 
        binary_preds = (all_preds > threshold).astype(int)
        conf_matrix = confusion_matrix(all_labels, binary_preds)
        test_auc_prc = average_precision_score(all_labels, all_preds)
        test_auc_roc = roc_auc_score(all_labels, all_preds)
    except:
        conf_matrix = [[0,0],[0,0]]
        test_auc_prc = -1
        test_auc_roc = -1

    return test_auc_prc, conf_matrix, test_auc_roc

def normal_test(X_train, y_train, X_test, y_test, X_val, y_val,max_epoch=500):
    X_train = X_train.to('cpu')
    X_test = X_test.to('cpu')
    X_val = X_val.to('cpu')

    y_train = y_train.reshape((-1, 1))
    y_val = y_val.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)

    mlp, best_val_auc = get_CLSmodel(X_train, y_train, X_val, y_val, max_epoch)

    test_auc_prc, conf_matrix, test_auc_roc = test_model(mlp, X_test, y_test)

    return test_auc_prc, conf_matrix, test_auc_roc, best_val_auc

def is_valid_smiles(smi):
    params = Chem.SmilesParserParams()
    params.removeHs = True
    mol = Chem.MolFromSmiles(smi, params)
    try:
        mol_tree = MolTree(smi)
    except:
        mol_tree = None
    return (mol is not None) and (mol_tree is not None)

origin_train_df = pd.read_csv(opts.origin_train_path)
gen_df = pd.read_csv(opts.gen_path)
test_df = pd.read_csv(opts.test_path)

origin_train_df = origin_train_df[origin_train_df['smiles'].apply(is_valid_smiles)].reset_index(drop=True)
gen_df = gen_df[gen_df['smiles'].apply(is_valid_smiles)].reset_index(drop=True)
test_df = test_df[test_df['smiles'].apply(is_valid_smiles)].reset_index(drop=True)

if aug:
    origin_train_df, val_df = train_test_split(origin_train_df , test_size = 0.25)
    combined_df = pd.concat([origin_train_df, gen_df], ignore_index=True)
    train_df = combined_df
    print(f"With aug--the size of generated data is {len(gen_df)}")
else:
    train_df, val_df = train_test_split(origin_train_df , test_size = 0.25)
    print(f"Without aug--the size of train data is {len(train_df)}")

print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}, Test set size: {len(test_df)}")

train_smiles = list(train_df['smiles'])
train_labels = np.array(train_df['label'])

val_smiles = list(val_df['smiles'])
val_labels = np.array(val_df['label'])

test_smiles = list(test_df['smiles'])
test_labels = np.array(test_df['label'])

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)

if opts.model_path is not None:
    model.load_state_dict(torch.load(opts.model_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

model = model.cuda()
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

MAX_EPOCH = 15
PRINT_ITER = 5

with open(save_path + '/log.txt', 'w+') as file:
    file.write("\n")

best_roc = 0
test_roc_on_best_val = 0
best_epoch = 0
for epoch in range(MAX_EPOCH):
    model.train()
    dataset = MoleculeDataset(train_smiles, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x, drop_last=True)
    word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0

    def replace_none_in_batch(batch):
        non_none_items = [item for item in batch if item[0] is not None and len(item[0].nodes)!=0]
        if not non_none_items:
            raise ValueError("All items in the batch are None")
        for i in range(len(batch)):
            if batch[i][0] is None or len(batch[i][0].nodes)==0:
                batch[i] = non_none_items[i % len(non_none_items)]
        return batch
    
    data_iter = iter(dataloader)
    for it in range(len(dataloader)):
        model.zero_grad()
        batch = next(data_iter)

        model.zero_grad()
        loss, kl_div, wacc, tacc, sacc, dacc, ploss = model(batch, is_pre = False, alpha = alpha, beta = beta)
        print("batch %d: loss: %.1f, KL: %.1f, Pro: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (it, loss, kl_div, ploss, wacc*100, tacc*100, sacc*100, dacc*100))
        loss.backward()
        optimizer.step()

        word_acc += wacc
        topo_acc += tacc
        assm_acc += sacc
        steo_acc += dacc

        if (it + 1) % PRINT_ITER == 0:
            word_acc = word_acc / PRINT_ITER * 100
            topo_acc = topo_acc / PRINT_ITER * 100
            assm_acc = assm_acc / PRINT_ITER * 100
            steo_acc = steo_acc / PRINT_ITER * 100

            print("PRINT_ITER 5, Loss, %.1f, KL: %.1f, Pro: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (loss, kl_div, ploss, word_acc, topo_acc, assm_acc, steo_acc))
            word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
            sys.stdout.flush()

        if (it + 1) % 15000 == 0:
            scheduler.step()
            try:
                print("learning rate: %.6f" % scheduler.get_last_lr()[0])
            except:
                pass

        if (it + 1) % 1000 == 0:
            torch.save(model.state_dict(), save_path + "/model.iter-%d-%d" % (epoch, it + 1))
            
    scheduler.step()

    torch.save(model.state_dict(), save_path + "/model.iter-" + str(epoch))

    valid_test_codes = []
    valid_test_label = []
    valid_train_codes = []
    valid_train_label = []
    valid_val_codes = []
    valid_val_label = []

    model.eval()
    with torch.no_grad():
        for i, item in enumerate(train_smiles):
            valid_train_codes.append(model.encode_latent_mean([item]))
            valid_train_label.append(train_labels[i])
        
        for i, item in enumerate(val_smiles):
            valid_val_codes.append(model.encode_latent_mean([item]))
            valid_val_label.append(val_labels[i])

        for i, item in enumerate(test_smiles):
            valid_test_codes.append(model.encode_latent_mean([item]))
            valid_test_label.append(test_labels[i])

    train_labels = np.array(valid_train_label)
    train_codes = torch.cat(valid_train_codes, dim=0)
    test_labels = np.array(valid_test_label)
    test_codes = torch.cat(valid_test_codes, dim=0)
    val_labels = np.array(valid_val_label)
    val_codes = torch.cat(valid_val_codes, dim=0)

    test_auc_prc_new, conf_matrix_new, test_auc_roc_new, best_val_auc = normal_test(train_codes, train_labels, test_codes, test_labels, val_codes, val_labels)
    print("val_roc_auc_score: ",best_val_auc)
    print("test_auc_roc_score_new: ",test_auc_roc_new)
    print("conf_matrix_new:")
    print(conf_matrix_new)

    if best_val_auc > best_roc:
        best_roc = best_val_auc
        best_epoch = epoch
        test_roc_on_best_val = test_auc_roc_new

        model_save_path = save_path +"/best_model"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path + rf"/best_model" + rf"/model.pt")

    with open(save_path + '/log.txt', 'a+') as file:
        file.write("\n")
        file.write("*"*8 + "\n")
        file.write("val_roc_auc_score: " + str(best_val_auc))
        file.write("\n")
        file.write("test_auc_roc_score_new: "+str(test_auc_roc_new))
        file.write("\n")
        file.write("conf_matrix_new:")
        file.write("\n")
        file.write(str(conf_matrix_new))
        file.write("\n")

        file.write("*"*8 + "\n")
        file.write("\n")

with open(opts.save_path + '/log.txt', 'a+') as file:
    file.write("\n")
    file.write("test_roc_on_best_val: "+str(test_roc_on_best_val)+' ')
    file.write("best_epoch: "+str(best_epoch))