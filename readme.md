This repo reproduces all the reuslts shown in our paper: https://arxiv.org/abs/2109.08191
Steps:

1) Run in the project dir: source ./init_project.sh



2) Generate the 'test-val' and 'test' subsets (as explained in the paper) for each dataset, by running:
python src/select_val_test_inds.py



3) Train DNNs for cifar10, cifar100, svhn, and tiny_imagenet using src/train.py.
For example, for CIFAR-10 run:
python src/train.py --dataset cifar10 --net resnet34 --checkpoint_dir /tmp/results/cifar10/resnet34/regular/resnet34_00 --epochs 300 --patience 3 --cooldown 0
And for the TRADES adversarial training run:
python src/train.py --dataset cifar10 --net resnet34 --checkpoint_dir /tmp/results/cifar10/resnet34/adv_robust_trades --epochs 300 --patience 3 --cooldown 0 --adv_trades True
If you wish also to reproduce results for the ensemble, train 9 more networks in:
/tmp/results/cifar10/resnet34/regular/resnet34_01
/tmp/results/cifar10/resnet34/regular/resnet34_02
/tmp/results/cifar10/resnet34/regular/resnet34_03
/tmp/results/cifar10/resnet34/regular/resnet34_04
/tmp/results/cifar10/resnet34/regular/resnet34_05
/tmp/results/cifar10/resnet34/regular/resnet34_06
/tmp/results/cifar10/resnet34/regular/resnet34_07
/tmp/results/cifar10/resnet34/regular/resnet34_08
/tmp/results/cifar10/resnet34/regular/resnet34_09
This repo assumes the above ordering for evaluation, for all the datasets.



4) For attacking a network, use src/attack.py.
For example, to attack CIFAR-10 with the FGSM^2 attack (defined in the paper), run:
python src/attack.py --checkpoint_dir /tmp/results/cifar10/resnet34/regular/resnet34_00 --attack fgsm
                     --attack_dir fgsm2 --eps 0.031
Attack both the regular DNNs and the adversarially trained DNNs for comparison.
To attack using the adaptive PGD attack (A-PGD) run:
python src/attack.py --checkpoint_dir /tmp/results/cifar10/resnet34/regular/resnet34_00 --attack whitebox_pgd
                     --attack_dir whitebox_pgd --eps 0.031 --eps_step 0.007 --max_iter 10 --tta_size 25
To attack using the adaptive FGSM attack (A-FGSM) run:
python src/attack.py --checkpoint_dir /tmp/results/cifar10/resnet34/regular/resnet34_00 --attack whitebox_pgd
                     --attack_dir whitebox_fgsm --eps 0.031 --eps_step 0.031 --max_iter 1 --tta_size 256
Note that running a PGD with a single step is equivalent to FGSM.



5) For evaluation use src/eval.py.
For example, for calculating the robustness using TTA on the previously attacked FGSM^2, run:
python src/eval.py --checkpoint_dir /tmp/results/cifar10/resnet34/regular/resnet34_00 --method tta --attack_dir fgsm2
The results will be written into the log file:
/tmp/results/cifar10/resnet34/regular/resnet34_00/fgsm2/tta/log.log
Note that for running with '--method random_forest' one must have in hand the TTAs for both the normal images and the
attack images. For our CIFAR-10 example above, generate TTAs for the normal images:
python src/eval.py --checkpoint_dir /tmp/results/cifar10/resnet34/regular/resnet34_00 --method tta --attack_dir ''
And only after that run:
python src/eval.py --checkpoint_dir /tmp/results/cifar10/resnet34/regular/resnet34_00 --method random_forest --attack_dir fgsm2



6) For the transferability results (the 'global' and 'LOOCV' setups in the supp mat), run with '--all_attacks' flag.
Also follow the instructions in src/eval.py (Line #132).
