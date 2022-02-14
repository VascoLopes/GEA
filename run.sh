# GEA
## nasbench201
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_1-096897.pth --n_runs 25 --dataset cifar10 --trainval >> results/new?/gea_nasbench201_cifar10.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_1-096897.pth --n_runs 25 --dataset cifar100 --trainval >> results/new?/gea_nasbench201_cifar100.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_1-096897.pth --n_runs 25 --dataset ImageNet16-120 --data_loc ../cifardata/ImageNet16-120/ --trainval >> results/new?/gea_nasbench201_imgnet16120.txt

## nasbench101
python3 gea.py --nasspace nasbench101 --api_loc ~/nas_benchmark_datasets/nasbench_full.tfrecord --n_runs 25 --dataset cifar10 --trainval >> results/new?/gea_nasbench101_cifar10.txt



# Ablation Studies
## S
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --sampling S --S 1 >> results/gea_nasbench201_cifar10_s1.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --sampling S --S 3 >> results/gea_nasbench201_cifar10_s3.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --sampling S --S 5 >> results/gea_nasbench201_cifar10_s5.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --sampling S --S 7 >> results/gea_nasbench201_cifar10_s7.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --sampling S --S 10 >> results/gea_nasbench201_cifar10_s10.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --sampling highest >> results/gea_nasbench201_cifar10_samplinghighest.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --sampling lowest >> results/gea_nasbench201_cifar10_samplinglowest.txt

## P
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --P 1 >> results/gea_nasbench201_cifar10_p1.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --P 3 >> results/gea_nasbench201_cifar10_p3.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --P 5 >> results/gea_nasbench201_cifar10_p5.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --P 7 >> results/gea_nasbench201_cifar10_p7.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --P 10 >> results/gea_nasbench201_cifar10_p10.txt

## Regularization
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --regularize oldest >> results/gea_nasbench201_cifar10_regularize_oldest.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --regularize highest >> results/gea_nasbench201_cifar10_regularize_highest.txt
python3 gea.py --nasspace nasbench201 --api_loc ~/nas_benchmark_datasets/NAS-Bench-201-v1_0-e61699.pth --n_runs 5 --dataset cifar10 --trainval --regularize lowest >> results/gea_nasbench201_cifar10_regularize_lowest.txt




