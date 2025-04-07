python3 ./main_USV.py --cv_run 5 --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr 0.8 --stiefel_opt mean --epochs 120 --run_name 'vgg16_cifar10_mean_cr0.8' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006
python3 ./main_USV.py --cv_run 5 --scheduler steplr --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr 0.5 --stiefel_opt mean --epochs 120 --run_name 'vgg16_cifar10_mean_cr0.5' --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006

python3 ./main_USV.py --cv_run 5 --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr 0.8 --stiefel_opt approx_orth --eps 1.1 --epochs 120 --run_name 'vgg16_cifar10_approx_eps1.1_cr0.8' --scheduler steplr  --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006
python3 ./main_USV.py --cv_run 5 --scheduler steplr --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr 0.5 --stiefel_opt approx_orth --eps 1.1 --epochs 120 --run_name 'add_vgg16_cifar10_approx_eps1.1_cr0.5' --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006

python3 ./main_USV.py --cv_run 5 --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr 0.8 --stiefel_opt approx_orth --eps 1.5 --epochs 120 --run_name 'vgg16_cifar10_approx_eps1.5_cr0.8' --scheduler steplr  --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006
python3 ./main_USV.py --cv_run 5 --scheduler steplr --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr 0.5 --stiefel_opt approx_orth --eps 1.5 --epochs 120 --run_name 'vgg16_cifar10_approx_eps1.5_cr0.5' --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006

python3 ./main_USV.py --cv_run 5 --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr -1 --retraction_opt qr --stiefel_opt cayley_sgd --epochs 120 --run_name 'vgg16_cifar10_cayley_sgd_qr' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006
python3 ./main_USV.py --cv_run 5 --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr -1 --retraction_opt cayley --stiefel_opt cayley_sgd --epochs 120 --run_name 'vgg16_cifar10_cayley_sgd' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006

python3 ./main_USV.py --cv_run 5 --scheduler steplr --lr 0.05 --momentum 0.45 --dataset_name cifar10 --net_name vgg16 --cr -1 --epochs 120 --run_name 'vgg16_cifar10_baseline' --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.001 0.002 0.003 0.004 0.005 0.006






