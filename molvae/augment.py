import sys
sys.path.append(rf'/data-extend/liuxg/TasProp') # use your desired default path
sys.path.append(rf'/data-extend/liuxg/TasProp/jtnn') # use your desired default path

import pandas as pd
import numpy as np
from numpy import ndarray
from torch import Tensor
import torch
import pickle
import os
import func_timeout
from func_timeout import func_set_timeout
from optparse import OptionParser

from rdkit import Chem

from jtnn import JTNNVAE, Vocab, create_var, MolTree

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

def is_len_eq(*args):
    lengths = [len(obj) for obj in args]
    return len(set(lengths)) == 1, lengths[0]

def getCode(model: JTNNVAE, smileses:list, labels:ndarray, has_label = True):
    valid_smileses = []
    valid_labels = []
    valid_codes = []
    lose = 0

    with torch.no_grad():
        for i,item in enumerate(smileses):
            if has_label:
                try:
                    cur_vec = model.encode_latent_mean([item])
                    valid_codes.append(cur_vec)
                    valid_smileses.append(item)
                    valid_labels.append(labels[i])
                except Exception as e:
                    lose += 1
            else:
                try:
                    cur_vec = model.encode_latent_mean([item])
                    valid_codes.append(cur_vec)
                    valid_smileses.append(item)
                except Exception as e:
                    lose += 1                
    
    return torch.cat(valid_codes, dim=0), valid_smileses, np.array(valid_labels), lose

def is_valid_smiles(smi):
    params = Chem.SmilesParserParams()
    params.removeHs = True
    mol = Chem.MolFromSmiles(smi, params)
    try:
        mol_tree = MolTree(smi)
    except:
        mol_tree = None
    return (mol is not None) and (mol_tree is not None)

parser = OptionParser()
parser.add_option("-t", "--task", dest="task_name")
parser.add_option("-d", "--delta", dest="delta", default=3.0)
parser.add_option("-w", "--write_path", dest="write_path")
parser.add_option("--vocab", dest="vocab_path", help="vocab file path (txt)", default=r'use your desired default path')
parser.add_option("--model", dest="model_path", help="model state dict path", default=r'use your desired default path')
parser.add_option("--train_path", dest="train_path", help="input train csv path (smiles,label)")
opts,args = parser.parse_args()

task_name = opts.task_name
delta = float(opts.delta)
write_path = opts.write_path

if opts.vocab_path is not None:
    vocab_path = opts.vocab_path
if opts.model_path is not None:
    model_path = opts.model_path
if opts.train_path is not None:
    train_path = opts.train_path


train_df = pd.read_csv(train_path)
train_df = train_df[train_df['smiles'].apply(is_valid_smiles)].reset_index(drop=True)

train_smiles = train_df['smiles'].tolist()
train_labels = train_df['label'].values

hidden_size = 450
latent_size = 56
depth = 3

vocab = [x.strip("\r\n") for x in open(vocab_path)]
vocab = Vocab(vocab)
jtnn = JTNNVAE(vocab, hidden_size, latent_size, depth)
jtnn.load_state_dict(torch.load(model_path))
if torch.cuda.is_available():
     jtnn = jtnn.cuda()

@func_set_timeout(5)
def decode(tree_vec, mol_vec, prob_decode):
    return jtnn.decode(tree_vec, mol_vec, prob_decode)

def neighborhood_molecule(cur_vec, delta=delta):
    with torch.no_grad():
        cur_vec = cur_vec.cuda().detach()
        mean_noise, var_noise = torch.zeros_like(cur_vec).detach(), torch.ones_like(cur_vec).detach()
        scale = torch.cat((torch.linspace(0, 0.1, 20), torch.linspace(0.1, 1, 20)), 0).detach()
        ms = set()

        for k in range(len(scale)):
            xy_range_random1 = np.random.choice(torch.linspace(-20,20,5))
            xy_range_random2 = np.random.choice(torch.linspace(-20,20,5))
            noise1 = torch.normal(mean_noise, var_noise).cuda().detach()
            noise2 = torch.normal(mean_noise, var_noise).cuda().detach()
            noise = (noise1 * xy_range_random1 + noise2 * xy_range_random2)*scale[k].detach()
            noise = noise.reshape((1,-1)).detach()
            all_vec = cur_vec.reshape((1,-1)).detach()

            tree_vec, mol_vec = np.hsplit(all_vec.detach().cpu().numpy(), 2)
            noise_tree, noise_mol = np.hsplit(noise.detach().cpu().numpy(), 2)

            tree_vec = create_var(torch.from_numpy(tree_vec).float() + torch.from_numpy(noise_tree).float()).detach()
            mol_vec = create_var(torch.from_numpy(mol_vec).float() + torch.from_numpy(noise_mol).float()).detach()

            neighbor_vec = torch.cat([tree_vec.cuda().detach(), mol_vec.cuda().detach()], dim=1) 
            distance = torch.dist(all_vec.cuda().detach(), neighbor_vec.cuda().detach(), p=2)
            if distance > delta:
                continue
            try:
                s = decode(tree_vec.detach(), mol_vec.detach(), prob_decode=False)
                if (s not in ms) and (s is not None):
                    ms.add(s)
                    print(s, distance)
            except func_timeout.exceptions.FunctionTimedOut:
                print('timeout')
                continue   
    return ms


valid_train_code, valid_train_smiles, valid_train_labels, lose = getCode(jtnn, train_smiles, train_labels, has_label=True)
assert is_len_eq(valid_train_code, valid_train_smiles, valid_train_labels)[0]

total = len(train_smiles)
step = max(1, total // 10)
generate_molecule_smiles_set = set()
with torch.no_grad():
    print("Total: ", total)
    for i in range(0, total):
        if i % step == 0:
            percent = int(i / total * 100)
            print(f"{percent}%")
        generate_molecule_smiles_set = generate_molecule_smiles_set.union(neighborhood_molecule(valid_train_code[i], delta=delta))

print("100%")

generate_molecule_smiles_list = list(generate_molecule_smiles_set - set(train_smiles))
valid_gen_code, valid_gen_smiles, _ , loss =  getCode(jtnn, generate_molecule_smiles_list, np.array([]), has_label=False)

dist_matrix = torch.cdist(valid_gen_code, valid_train_code, p=2)
min_dists = torch.min(dist_matrix, 1).values
min_dist_indices = torch.min(dist_matrix, 1).indices
label_copy = np.array(valid_train_labels)
gen_labels = list(label_copy[min_dist_indices.cpu().numpy()])

invalid_indices = torch.nonzero((min_dists > delta)==True).squeeze()
whole_indices = torch.tensor(range(len(valid_gen_code)), dtype=invalid_indices.dtype).cuda()
valid_indices = whole_indices[~np.isin(whole_indices.cpu(), invalid_indices.cpu())]
valid_indices = valid_indices.sort().values

final_gen_smiles = list(np.array(valid_gen_smiles)[list(valid_indices.cpu().numpy())])
final_gen_labels = list(np.array(gen_labels)[list(valid_indices.cpu().numpy())])

gen_df = pd.DataFrame({'smiles':final_gen_smiles, 'label':final_gen_labels})

os.makedirs(os.path.dirname(write_path), exist_ok=True)
gen_df.to_csv(write_path, index=False)