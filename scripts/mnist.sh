python3 ./main_USV.py --cv_run 5 --lr 0.1 --momentum 0.3 --dataset_name mnist --net_name lenet5 --cr 0.8 --stiefel_opt mean --epochs 120 --run_name 'lenet_mnist_mean_cr0.8' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06

python3 ./main_USV.py --cv_run 5 --lr 0.1 --momentum 0.3 --dataset_name mnist --net_name lenet5 --cr 0.8 --stiefel_opt approx_orth --eps 1.1 --epochs 120 --run_name 'lenet_mnist_approx1.1_cr0.8' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06

python3 ./main_USV.py --cv_run 5  --lr 0.1 --momentum 0.3 --dataset_name mnist --net_name lenet5 --cr 0.8 --stiefel_opt approx_orth --eps 1.5 --epochs 120 --run_name 'lenet_mnist_approx1.5_cr0.8' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06

python3 ./main_USV.py --cv_run 5 --lr 0.1 --momentum 0.3 --dataset_name mnist --net_name lenet5 --cr 0.5 --stiefel_opt mean --epochs 120 --run_name 'lenet_mnist_mean_cr0.5' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06
python3 ./main_USV.py --cv_run 5 --lr 0.1 --momentum 0.3 --dataset_name mnist --net_name lenet5 --cr 0.5 --stiefel_opt approx_orth --eps 1.1 --epochs 120 --run_name 'lenet_mnist_approx1.1_cr0.5' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06
python3 ./main_USV.py --cv_run 5 --lr 0.1 --momentum 0.3 --dataset_name mnist --net_name lenet5 --cr 0.5 --stiefel_opt approx_orth --eps 1.5 --epochs 120 --run_name 'lenet_mnist_approx1.5_cr0.5' --scheduler steplr --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06

python3 ./main_USV.py --cv_run 5 --lr 0.1 --momentum 0.3 --dataset_name mnist --scheduler steplr --net_name lenet5 --cr -1 --retraction_opt cayley --stiefel_opt cayley_sgd --epochs 120 --run_name 'lenet_mnist_cayley_sgd' --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06

python3 ./main_USV.py --cv_run 5 --lr 0.1 --momentum 0.3 --dataset_name mnist --scheduler steplr --net_name lenet5 --cr -1 --retraction_opt qr --stiefel_opt cayley_sgd --epochs 120 --run_name 'lenet_mnist_sgd_qr' --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06

python3 ./main_USV.py --cv_run 5 --scheduler steplr --lr 0.1 --momentum 0.3 --dataset_name mnist --net_name lenet5 --cr -1 --epochs 120 --run_name 'lenet_mnist_baseline' --save_progress 1 --save_weights 1 --attack_name fgsm --p_budget 0 0.01 0.02 0.03 0.04 0.05 0.06
