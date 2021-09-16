'''Attack networks with PyTorch and ART'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import json
import os
import argparse
import logging
from cleverhans.utils import random_targets, to_categorical
import sys
sys.path.insert(0, "./adversarial_robustness_toolbox")

from src.datasets.train_val_test_data_loaders import get_test_loader, get_normalized_tensor
from src.datasets.utils import get_mini_dataset_inds
from src.datasets.tta_utils import get_tta_transforms
from src.utils import boolean_string, set_logger, get_image_shape
from src.models.utils import get_strides, get_conv1_params, get_model
from src.attacks.tta_whitebox_projected_gradient_descent import TTAWhiteboxProjectedGradientDescent
from src.classifiers.pytorch_tta_classifier import PyTorchTTAClassifier

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, SaliencyMapMethod, \
    CarliniL2Method, CarliniLInfMethod

parser = argparse.ArgumentParser(description='Attacking networks using pytorch')
parser.add_argument('--checkpoint_dir', default='/tmp/results/cifar10/resnet34/regular/resnet34_00', type=str, help='checkpoint dir')
parser.add_argument('--checkpoint_file', default='ckpt.pth', type=str, help='checkpoint path file name')
parser.add_argument('--attack', default='fgsm', type=str, help='attack: fgsm, jsma, pgd, deepfool, cw_L2, cw_Linf, whitebox_pgd')
parser.add_argument('--attack_dir', default='fgsm2', type=str, help='attack directory')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='Data loading threads')

# for fgsm/pgd/cw_Linf/whitebox_pgd:
parser.add_argument('--eps'     , default=0.031, type=float, help='maximum Linf deviation from original image')
parser.add_argument('--eps_step', default=0.003, type=float, help='step size of each adv iteration')

# for whitebox_pgd:
parser.add_argument('--max_iter', default=10, type=int, help='Number of TTAs to use in the PGD whitebox attack')
parser.add_argument('--tta_size', default=25, type=int, help='Number of TTAs to use in the PGD whitebox attack')

args = parser.parse_args()

# to reproduce bit exact attack:
# torch.manual_seed(9)
# random.seed(9)
# np.random.seed(9)

args.targeted = args.attack != 'deepfool'  # whether or not to use a targeted attack.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'r') as f:
    train_args = json.load(f)

CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, args.checkpoint_file)
if args.attack_dir != '':
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack_dir)
else:
    ATTACK_DIR = os.path.join(args.checkpoint_dir, args.attack)
os.makedirs(ATTACK_DIR, exist_ok=True)
batch_size = args.batch_size

log_file = os.path.join(ATTACK_DIR, 'log.log')
set_logger(log_file)
logger = logging.getLogger()
# rand_gen = np.random.RandomState(seed=12345)

dataset = train_args['dataset']
_, test_inds = get_mini_dataset_inds(dataset)
test_size = len(test_inds)

# Data
logger.info('==> Preparing data..')
testloader = get_test_loader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)
img_shape = get_image_shape(dataset)
classes = testloader.dataset.classes
all_test_size  = len(testloader.dataset)

# Model
logger.info('==> Building model..')
conv1 = get_conv1_params(dataset)
strides = get_strides(dataset)
global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
if 'best_net' in global_state:  # this is the default for ckpt.pth. For ckpt_epoch_100.pth (for example) we stay with the global_state
    global_state = global_state['best_net']
net = get_model(train_args['net'])(num_classes=len(classes), activation=train_args['activation'], conv1=conv1, strides=strides)
net = net.to(device)
net.load_state_dict(global_state)
net.eval()
# summary(net, (img_shape[2], img_shape[0], img_shape[1]))
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=train_args['lr'],
    momentum=train_args['mom'],
    weight_decay=0.0,  # train_args['wd'],
    nesterov=train_args['mom'] > 0)

X_test = get_normalized_tensor(testloader, img_shape, batch_size)
y_test = np.asarray(testloader.dataset.targets)

classifier = PyTorchTTAClassifier(model=net, clip_values=(0, 1), loss=criterion,
                                  optimizer=optimizer, input_shape=(img_shape[2], img_shape[0], img_shape[1]), nb_classes=len(classes))

y_test_logits = classifier.predict(X_test, batch_size=batch_size)
y_test_preds = y_test_logits.argmax(axis=1)
test_acc = np.sum(y_test_preds == y_test) / all_test_size
logger.info('Accuracy on benign test examples: {}%'.format(test_acc * 100))

# attack
# creating targeted labels
if args.targeted:
    tgt_file = os.path.join(ATTACK_DIR, 'y_test_adv.npy')
    if not os.path.isfile(tgt_file):
        y_test_targets = random_targets(y_test, len(classes))
        y_test_adv = y_test_targets.argmax(axis=1)
        np.save(os.path.join(ATTACK_DIR, 'y_test_adv.npy'), y_test_adv)
    else:
        y_test_adv = np.load(os.path.join(ATTACK_DIR, 'y_test_adv.npy'))
        y_test_targets = to_categorical(y_test_adv, nb_classes=len(classes))
else:
    y_test_adv = None
    y_test_targets = None

if args.attack == 'fgsm':
    attack = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=args.eps,
        eps_step=args.eps_step,
        targeted=args.targeted,
        num_random_init=0,
        batch_size=batch_size
    )
elif args.attack == 'pgd':
    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=args.eps,
        eps_step=args.eps_step,
        targeted=args.targeted,
        batch_size=batch_size
    )
elif args.attack == 'whitebox_pgd':
    attack = TTAWhiteboxProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=args.eps,
        eps_step=args.eps_step,
        max_iter=args.max_iter,
        targeted=args.targeted,
        batch_size=batch_size,
        tta_transforms=get_tta_transforms(dataset)
    )
elif args.attack == 'deepfool':
    attack = DeepFool(
        classifier=classifier,
        max_iter=50,
        epsilon=0.02,
        nb_grads=len(classes),
        batch_size=batch_size
    )
elif args.attack == 'jsma':
    attack = SaliencyMapMethod(
        classifier=classifier,
        theta=1.0,
        gamma=0.01,
        batch_size=batch_size
    )
elif args.attack == 'cw_L2':
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=0.8,
        targeted=args.targeted,
        initial_const=0.1,
        batch_size=batch_size
    )
elif args.attack == 'cw_Linf':
    attack = CarliniLInfMethod(
        classifier=classifier,
        confidence=0.8,
        targeted=args.targeted,
        batch_size=batch_size,
        eps=args.eps
    )
else:
    err_str = 'Attack {} is not supported'.format(args.attack)
    logger.error(err_str)
    raise AssertionError(err_str)

dump_args = args.__dict__.copy()
dump_args['attack_params'] = {}
for param in attack.attack_params:
    if param in attack.__dict__.keys():
        dump_args['attack_params'][param] = attack.__dict__[param]
with open(os.path.join(ATTACK_DIR, 'attack_args.txt'), 'w') as f:
    json.dump(dump_args, f, indent=2)

if not os.path.exists(os.path.join(ATTACK_DIR, 'X_test_adv.npy')):
    X_test_adv = attack.generate(x=X_test, y=y_test_targets)
    test_adv_logits = classifier.predict(X_test_adv, batch_size=batch_size)
    y_test_adv_preds = np.argmax(test_adv_logits, axis=1)
    np.save(os.path.join(ATTACK_DIR, 'X_test_adv.npy'), X_test_adv)
    np.save(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'), y_test_adv_preds)
else:
    X_test_adv       = np.load(os.path.join(ATTACK_DIR, 'X_test_adv.npy'))
    y_test_adv_preds = np.load(os.path.join(ATTACK_DIR, 'y_test_adv_preds.npy'))

test_adv_accuracy = np.mean(y_test_adv_preds == y_test)
logger.info('Accuracy on adversarial test examples: {}%'.format(test_adv_accuracy * 100))
logger.handlers[0].flush()
