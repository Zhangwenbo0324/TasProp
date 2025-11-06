from torch.utils.data import Dataset
from mol_tree import MolTree
import numpy as np
import pickle
import sys

class MoleculeDataset(Dataset):

    def __init__(self, smiles_file, labels_file=None):
        if type(smiles_file) is str:
            with open(smiles_file) as f:
                self.smiles = [line.strip("\r\n").split()[0] for line in f]

            print("len: ", len(self.smiles))
            if labels_file is not None:
                with open(labels_file, 'rb') as f: 
                    self.labels = pickle.load(f)
            else:
                self.labels=None
        elif type(smiles_file) is list:
            self.smiles = smiles_file
            self.labels = labels_file
        else:
            raise Exception('Wrong type!')



    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            try:
                mol_smiles = self.smiles[idx]
                mol_label = self.labels[idx]
                mol_tree = MolTree(mol_smiles)
                mol_tree.recover()
                mol_tree.assemble()
                return [mol_tree, mol_label]
            except:
                return [None, None]
        else:
            try:
                mol_smiles = self.smiles[idx]
                mol_tree = MolTree(mol_smiles)
                # if mol_tree.isFail:
                #     return None
                mol_tree.recover()
                mol_tree.assemble()
            except:
                return None
            return mol_tree


class PropDataset(Dataset):

    def __init__(self, data_file, prop_file):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]

