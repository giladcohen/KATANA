"""Reproducing the plot of the ablation study of tta_size (Figure 3 in the paper and Figure G1 in the supp mat), for
 datasets: CIFAR10, CIFAR100, SVHN,
 attack: PDG, Deepfool, and CW_L2
"""

import os
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

def attack_to_dir(attack: str):
    if attack == 'PGD':
        return 'pgd1'
    elif attack == 'Deepfool':
        return 'deepfool'
    elif attack == 'CW':
        return 'cw_L2'

def get_acc_from_log(file: str):
    acc = np.nan
    with open(file, 'r') as f:
        for line in f:
            if 'INFO Test accuracy:' in line:
                acc = float(line.split('accuracy: ')[1].split('%')[0])
    assert acc is not None
    return acc


num_experiments = 5
CHECKPOINT_ROOT = '/tmp/results'
datasets = ['cifar10', 'cifar100', 'svhn']
attacks = ['PGD', 'Deepfool', 'CW']
tta_size_vec = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
tta_size_ext_vec = []
data_ext = {}
for size in tta_size_vec:
    tta_size_ext_vec.extend([size] * num_experiments)

data = {}
for dataset in datasets:
    data[dataset] = {}
    for attack in attacks:
        data[dataset][attack] = {}
        for tta_size in tta_size_vec:
            data[dataset][attack][tta_size] = []
            for n in range(1, num_experiments + 1):
                file = os.path.join(CHECKPOINT_ROOT, dataset, 'resnet34', 'regular', 'resnet34_00', attack_to_dir(attack), 'tta_size_{}_exp{}'.format(tta_size, n), 'log.log')
                acc = get_acc_from_log(file)
                acc = np.round(acc, 2)
                data[dataset][attack][tta_size].append(acc)
            data[dataset][attack][tta_size] = np.asarray(data[dataset][attack][tta_size])

for dataset in datasets:
    plt.close()
    for attack in attacks:
        data_ext[attack] = []
        for size in tta_size_vec:
            data_ext[attack].extend(data[dataset][attack][size])

    d = {'tta_size': tta_size_ext_vec}
    for attack in attacks:
        d.update({attack: data_ext[attack]})
    df = pd.DataFrame(d)

    if dataset == 'cifar10':
        dataset_str = 'CIFAR-10'
    elif dataset == 'cifar100':
        dataset_str = 'CIFAR-100'
    else:
        dataset_str = 'SVHN'

    g = sns.lineplot(x='tta_size', y='value', hue='variable', data=pd.melt(df, ['tta_size']), ci='sd',
                     palette='husl', style='variable')
    g.set(xscale='log', xlabel='TTA size', ylabel='Accuracy [%]', title=dataset_str)
    g.legend_.set_title('Attack')
    g.set_xticks(tta_size_vec)
    g.set_xticklabels(tta_size_vec)
    plt.show()
    # plt.savefig('tta_size_ablation_{}.png'.format(dataset), dpi=300)
