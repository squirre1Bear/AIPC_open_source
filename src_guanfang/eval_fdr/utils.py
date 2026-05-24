import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import Tensor
from torch import nn

PROTON_MASS_AMU = 1.007276


def get_fdr_result(df):
    df = df.sort_values(by='score', ascending=False, ignore_index=True)
    df['decoy'] = np.where(df['label'] == 1, 0, 1)

    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    df['q_value'] = decoy_num / target_num
    df['q_value'] = df['q_value'][::-1].cummin()

    #  conservative FDR estimate
    return df[['scan_number', 'precursor_mz', 'precursor_charge', 'modified_sequence',
               'score', 'label',  'q_value', 'file_name']]

def padding(data):
    ll = torch.tensor([x.shape[0] for x in data], dtype=torch.long)
    data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    data_mask = torch.arange(data.shape[1], dtype=torch.long)[None, :] >= ll[:, None]
    return data, data_mask

def collate_batch(batch):
    """Collate batch of samples."""
    spectrum, precursor_mzs, precursor_charges, peptides, tokens, label, index = zip(*batch)
    # spectrum, precursor_mzs, precursor_charges, peptides, tokens, label, index = zip(*batch)    # edit_by_zxf_20241104

    # Pad spectra
    spectra, spectra_mask = padding(spectrum)

    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()

    # deltaRT = torch.tensor(deltaRT)
    # precursors = torch.vstack([precursor_masses, precursor_charges, deltaRT]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    index = torch.tensor(index).to(torch.float)
    # lowHz = torch.tensor(lowHz).to(torch.float)    # edit_by_zxf_20241104
    return spectra, spectra_mask, precursors, tokens, label, index
    # return spectra, spectra_mask, precursors, tokens, label, index, lowHz    # edit_by_zxf_20241104
