'''Test robustness with PyTorch.'''
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import logging
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import sys
sys.path.insert(0, "./adversarial_robustness_toolbox")

from src.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from src.datasets.tta_utils import get_tta_logits
from src.datasets.utils import get_mini_dataset_inds, get_ensemble_dir, get_dump_dir
from src.utils import set_logger, get_ensemble_paths, majority_vote, convert_tensor_to_image, \
    print_Linf_dists, calc_attack_rate, get_image_shape
from src.models.utils import get_strides, get_conv1_params, get_model

from art.classifiers import PyTorchClassifier

parser = argparse.ArgumentParser(description='Evaluating robustness score')
parser.add_argument('--checkpoint_dir', default='/tmp/results/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--method', default='tta', type=str, help='simple, ensemble, tta, random_forest, logistic_regression, svm_linear, svm_rbf')
parser.add_argument('--attack_dir', default='fgsm2', type=str, help='attack directory, or None for normal images')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# tta method params:
parser.add_argument('--tta_size', default=256, type=int, help='number of test-time augmentations')
parser.add_argument('--gaussian_std', default=0.005, type=float, help='Standard deviation of Gaussian noise')
parser.add_argument('--tta_output_dir', default='tta', type=str, help='The dir to dump the tta results for further use')
parser.add_argument('--soft_transforms', action='store_true', help='applying mellow transforms')
parser.add_argument('--clip_inputs', action='store_true', help='clipping TTA inputs between 0 and 1')
parser.add_argument('--overwrite', action='store_true', help='force calculating and saving TTA')
parser.add_argument('--num_workers', default=20, type=int, help='Data loading threads for tta loader or random forest')

# random forest params
parser.add_argument('--all_attacks', action='store_true',
                    help='Train random forest for all attacks (global) or for Leave-One-Out Cross Validation methods')

# dump
parser.add_argument('--dump_dir', default=None, type=str, help='dump dir for logs and data')

args = parser.parse_args()
if args.dump_dir is None:
    args.dump_dir = args.method

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)
is_attacked = args.attack_dir != ''

if args.all_attacks:
    ATTACK_DIR = os.path.join(args.checkpoint_dir, 'all_attacks')
    attack_args = None
    targeted = None
    y_adv = None
elif is_attacked:
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
    with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'r') as f:
        attack_args = json.load(f)
    targeted = attack_args['targeted']
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
batch_size = args.batch_size

if args.all_attacks:
    assert args.dump_dir is not None
    DUMP_DIR = os.path.join(ATTACK_DIR, args.dump_dir)
else:
    DUMP_DIR = get_dump_dir(args.checkpoint_dir, args.dump_dir, args.attack_dir)
os.makedirs(DUMP_DIR, exist_ok=True)
log_file = os.path.join(DUMP_DIR, 'log.log')

# dumping args to txt file
with open(os.path.join(DUMP_DIR, 'eval_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

set_logger(log_file)
logger = logging.getLogger()
rand_gen = np.random.RandomState(seed=12345)

dataset = train_args['dataset']
val_inds, test_inds = get_mini_dataset_inds(dataset)
val_size = len(val_inds)
test_size = len(test_inds)

# get data:
test_loader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=device=='cuda')
img_shape = get_image_shape(dataset)
X_test = get_normalized_tensor(test_loader, img_shape, batch_size)
y_test = np.asarray(test_loader.dataset.targets)
classes = test_loader.dataset.classes

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:
    global_state = global_state['best_net']
net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'], conv1=conv1, strides=strides)
net = net.to(device)
net.load_state_dict(global_state)
net.eval()  # frozen
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# setting classifier
classifier = PyTorchClassifier(model=net, clip_values=(0, 1), loss=None,
                               optimizer=None, input_shape=(img_shape[2], img_shape[0], img_shape[1]), nb_classes=len(classes))

y_gt = y_test[test_inds]
# y_orig_norm_preds = pytorch_evaluate(net, test_loader, ['probs'])[0].argmax(axis=1)[test_inds]
y_orig_norm_preds = classifier.predict(X_test[test_inds], batch_size).argmax(axis=1)
orig_norm_acc = np.mean(y_orig_norm_preds == y_gt)
logger.info('Normal test accuracy: {}%'.format(100 * orig_norm_acc))

if args.method == 'random_forest' and args.all_attacks:
    del test_loader, X_test, net, classifier
    # run with  '--all_attacks True' if you wish to reproduce the "global" and "LOOCV" robustness as presented in
    # Table F1 in the supplementary material.
    # For "global" robustness, leave all lines.
    # For LOOCV, comment out the relevant attacks as portrayed on Table F2. The below example shows the settings for
    # LOOCV when calculatiing robustness for CW_L2 and CW_Linf.

    attack_train_set = [
        'fgsm1',
        'fgsm2',
        'jsma',
        'pgd1',
        'pgd2',
        'deepfool',
        # 'cw_L2',
        # 'cw_Linf'
    ]
    attack_test_set = [
        # 'fgsm1',
        # 'fgsm2',
        # 'jsma',
        # 'pgd1',
        # 'pgd2',
        # 'deepfool',
        'cw_L2',
        'cw_Linf'
    ]
    # load normal tta_logits:
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, '')
    tta_logits_norm = np.load(os.path.join(tta_dir, 'tta_logits.npy'))
    tta_logits_train_norm, tta_logits_test_norm = tta_logits_norm[val_inds], tta_logits_norm[test_inds]

    # load attacked tta_logits to train:
    tta_logits_train_adv = []
    for attack_dir in attack_train_set:
        tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, attack_dir)
        tta_logits_adv = np.load(os.path.join(tta_dir, 'tta_logits.npy'))
        tta_logits_train_adv.append(tta_logits_adv[val_inds])
    tta_logits_train_adv = np.vstack(tta_logits_train_adv)

    # load attacked tta_logits to test:
    tta_logits_test_adv = []
    for attack_dir in attack_test_set:
        tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, attack_dir)
        tta_logits_adv = np.load(os.path.join(tta_dir, 'tta_logits.npy'))
        tta_logits_test_adv.append(tta_logits_adv[test_inds])
    tta_logits_test_adv = np.vstack(tta_logits_test_adv)

    # reshape to features:
    features_train_norm = tta_logits_train_norm.reshape((val_size, -1))
    features_test_norm = tta_logits_test_norm.reshape((test_size, -1))
    features_train_adv = tta_logits_train_adv.reshape((len(attack_train_set) * val_size, -1))
    features_test_adv = tta_logits_test_adv.reshape((len(attack_test_set) * test_size, -1))

    # concatenate features:
    features_train = np.concatenate((features_train_norm, features_train_adv), axis=0)
    labels_train = np.tile(y_test[val_inds], len(attack_train_set) + 1)

    # set sample weights:
    sample_weights = None
    # sample_weights = []
    # sample_weights.append([1.0] * val_size)  # for normal
    # for attack, weight in attack_set.items():
    #     sample_weights.append([weight] * val_size)
    # sample_weights = np.hstack(sample_weights)

    logger.info('Initializing random forest classifier for all attacks...')
    clf = RandomForestClassifier(
        n_estimators=1000,  # for debug
        criterion="gini",  # gini or entropy
        max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
        # until all leaves contain less than min_samples_split samples.
        bootstrap=True, # Whether bootstrap samples are used when building trees.
        # If False, the whole dataset is used to build each tree.
        random_state=rand_gen,
        verbose=1000,
        n_jobs=args.num_workers
    )
    logger.info('Start training the classifier...')
    clf.fit(features_train, labels_train, sample_weights)
    logger.info('Predicting normal samples with the classifier...')
    y_preds_norm = clf.predict(features_test_norm)
    logger.info('Predicting adversarial samples with the classifier...')
    y_preds_all_attacks = clf.predict(features_test_adv)

    # print accuracies
    all_acc = {}
    all_acc['normal'] = np.round(np.mean(y_preds_norm == y_gt) * 100.0, 2)
    cnt = 0
    for attack in attack_test_set:
        y_preds = y_preds_all_attacks[cnt*test_size:(cnt+1)*test_size]
        all_acc[attack] = np.round(np.mean(y_preds == y_gt) * 100.0, 2)
        cnt += 1
    logger.info('Test accuracy on normal and all attacks: {}'.format(all_acc))
    logger.handlers[0].flush()
    exit(0)
elif not is_attacked:
    if args.method == 'simple':
        logger.info('done')  # already calculated above
        exit(0)
    else:
        logger.info('considering original images only...')
        X = X_test
else:
    logger.info('considering adv images of attack {}. targeted={}'.format(attack_args['attack'], attack_args['targeted']))
    X = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
    y_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy')) if attack_args['targeted'] else None
    print_Linf_dists(X[test_inds], X_test[test_inds])

if args.method == 'simple':
    y_preds = classifier.predict(X[test_inds], batch_size).argmax(axis=1)
elif args.method == 'ensemble':
    ensemble_dir = get_ensemble_dir(dataset, train_args['net'])
    networks_list = get_ensemble_paths(ensemble_dir)
    networks_list.remove(CHECKPOINT_PATH)
    num_networks = len(networks_list)
    y_preds_nets = np.nan * np.ones((test_size, num_networks), dtype=np.int32)
    for j, ckpt_file in tqdm(enumerate(networks_list)):  # for network j
        logger.info('Evaluating network {}'.format(ckpt_file))
        global_state = torch.load(ckpt_file, map_location=torch.device(device))
        net.load_state_dict(global_state['best_net'])
        y_preds_nets[:, j] = classifier.predict(X[test_inds], batch_size=batch_size).argmax(axis=1)
    assert not np.isnan(y_preds_nets).any()
    y_preds_nets = y_preds_nets.astype(np.int32)
    y_preds = np.apply_along_axis(majority_vote, axis=1, arr=y_preds_nets)
elif args.method == 'tta':
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, args.attack_dir)
    os.makedirs(tta_dir, exist_ok=True)
    tta_file = os.path.join(tta_dir, 'tta_logits.npy')
    if args.overwrite:
        logger.info('Calculating tta logits (overwrite). It will take couple of minutes...')
        tta_logits = get_tta_logits(dataset, args, net, X, y_test, args.tta_size, len(classes))
        np.save(os.path.join(tta_dir, 'tta_logits.npy'), tta_logits)
    else:
        try:
            logger.info('Try loading tta logits from {}...'.format(tta_file))
            tta_logits = np.load(tta_file)
        except Exception as e:
            logger.warning('Did not load tta logits from {}. Exception err: {}'.format(tta_file, e))
            logger.info('Calculating tta logits. It will take couple of minutes...')
            tta_logits = get_tta_logits(dataset, args, net, X, y_test, args.tta_size, len(classes))
            np.save(os.path.join(tta_dir, 'tta_logits.npy'), tta_logits)

    # testing only test_inds:
    tta_logits = tta_logits[test_inds]
    # tta_probs = scipy.special.softmax(tta_logits, axis=2)
    # tta_preds = tta_probs.argmax(axis=2)
    # y_preds = np.apply_along_axis(majority_vote, axis=1, arr=tta_preds)
    y_preds = tta_logits.sum(axis=1).argmax(axis=1)

elif args.method in ['logistic_regression', 'svm_linear', 'svm_rbf', 'random_forest']:
    assert is_attacked, 'method {} can only be run with an attack'.format(args.method)

    # delete unnecessary memory for faster calculation
    del test_loader, X_test, net, classifier, X

    # load tta logits:
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, '')
    if os.path.exists(os.path.join(tta_dir, 'tta_logits.npy')):
        tta_logits_norm = np.load(os.path.join(tta_dir, 'tta_logits.npy'))
    else:
        raise AssertionError('TTA logits for normal images must be generated prior to running with method {}'.format(args.method))
    tta_dir = get_dump_dir(args.checkpoint_dir, args.tta_output_dir, args.attack_dir)
    if os.path.exists(os.path.join(tta_dir, 'tta_logits.npy')):
        tta_logits_adv = np.load(os.path.join(tta_dir, 'tta_logits.npy'))
    else:
        raise AssertionError('TTA logits for attacked images must be generated prior to running with method {}'.format(args.method))

    # reshape to features:
    tta_logits_train_norm, tta_logits_test_norm = tta_logits_norm[val_inds], tta_logits_norm[test_inds]
    tta_logits_train_adv, tta_logits_test_adv = tta_logits_adv[val_inds], tta_logits_adv[test_inds]

    features_train_norm = tta_logits_train_norm.reshape((val_size, -1))
    features_test_norm = tta_logits_test_norm.reshape((test_size, -1))
    features_train_adv = tta_logits_train_adv.reshape((val_size, -1))
    features_test_adv = tta_logits_test_adv.reshape((test_size, -1))

    # concatenate features:
    features_train = np.concatenate((features_train_norm, features_train_adv), axis=0)
    # features_test = np.concatenate((features_test_norm, features_test_adv), axis=0)
    labels_train = np.concatenate((y_test[val_inds], y_test[val_inds]), axis=0)
    # labels_test = np.concatenate((y_test[test_inds], y_test[test_inds]), axis=0)

    if args.method == 'logistic_regression':
        logger.info('Initializing logistic regression classifier...')
        clf = LogisticRegression(multi_class='ovr', random_state=rand_gen, n_jobs=args.num_workers, verbose=1)
    elif args.method == 'svm_linear':
        logger.info('Initializing linear SVM classifier...')
        clf = LinearSVC(penalty='l2', loss='hinge', multi_class='ovr', random_state=rand_gen, verbose=1)
    elif args.method == 'svm_rbf':
        logger.info('Initializing RBF SVM classifier...')
        clf = SVC(kernel='rbf', decision_function_shape='ovr', break_ties=True, random_state=rand_gen, verbose=1)
    elif args.method == 'random_forest':
        logger.info('Initializing random forest classifier...')
        clf = RandomForestClassifier(
            n_estimators=1000,
            criterion="gini",  # gini or entropy
            max_depth=None, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
            # until all leaves contain less than min_samples_split samples.
            bootstrap=True, # Whether bootstrap samples are used when building trees.
            # If False, the whole datset is used to build each tree.
            random_state=rand_gen,
            verbose=1000,
            n_jobs=args.num_workers
        )
    else:
        logger.error('How did we get here?')
        raise AssertionError

    logger.info('Start training the classifier...')
    clf.fit(features_train, labels_train)
    logger.info('Predicting normal samples with the classifier...')
    y_preds_norm = clf.predict(features_test_norm)
    logger.info('Predicting adversarial samples with the classifier...')
    y_preds = clf.predict(features_test_adv)

    acc = np.mean(y_preds_norm == y_gt)
    logger.info('New normal test accuracy of {}: {}%'.format(args.method, 100.0 * acc))
else:
    raise AssertionError('unknown method {}'.format(args.method))

# metrics calculation:
acc = np.mean(y_preds == y_gt)
logger.info('Test accuracy: {}%'.format(100.0 * acc))

if is_attacked:
    attack_rate = calc_attack_rate(y_preds, y_orig_norm_preds, y_gt)
    logger.info('attack success rate: {}%'.format(100.0 * attack_rate))

logger.handlers[0].flush()
