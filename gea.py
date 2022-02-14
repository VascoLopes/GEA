import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from tqdm import trange
from statistics import mean
import time
import collections
from utils import add_dropout
from operator import itemgetter


parser = argparse.ArgumentParser(description='GEA')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='~/nas_benchmark_datasets/NAS-Bench-201-v1_1-096897.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/', type=str, help='folder to save results')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--evaluate_size', default=256, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=1, type=int)
parser.add_argument('--regularize', default='oldest', type=str, help='which scheme to use to remove indvs from population',
                    choices=["oldest", "highest", "lowest"])
parser.add_argument('--sampling', default='S', type=str, help='which scheme to use to sample candidates to be parent',
                    choices=["S", "highest", "lowest"])
parser.add_argument('--C', default=200, type=int)
parser.add_argument('--P', default=10, type=int)
parser.add_argument('--S', default=5, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def get_batch_jacobian(net, x):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob#, grad


def eval_score_perclass(jacob, labels=None, n_classes=10):
    k = 1e-5
    #n_classes = len(np.unique(labels))
    per_class={}
    for i, label in enumerate(labels):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        s = 0
        try:
            corrs = np.corrcoef(per_class[c])

            s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
            if n_classes > 100:
                s /= len(corrs)
        except: # defensive programming
            continue

        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:

        for c in ind_corr_matrix_score_keys:
            # B)
            score += np.absolute(ind_corr_matrix_score[c])
    else: 
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

        # should divide by number of classes seen
        score /= len(ind_corr_matrix_score_keys)

    return score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)


times     = []
chosen    = []
acc       = []
val_acc   = []
topscores = []
order_fn = np.nanargmax


if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'


C = C_AUX = args.C #200
P = P_AUX = args.P
S = args.S

times = []
histories = []
best_arch = []
best_arch_val_acc = []
runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    scores = []

    npstate = np.random.get_state()
    ranstate = random.getstate()
    torchstate = torch.random.get_rng_state()
    
    init_population = []
    total_time_cost = 0
    start = time.time()

    data_iterator = iter(train_loader)
    x, target = next(data_iterator)
    x, target = x.to(device), target.cpu().tolist()
    while len(init_population) < C:
        uid = searchspace.random_arch() # gt random id for arch
        #print(uid)
        uid = searchspace[uid] # for nasbench101 hash
        #print(uid)
        network = searchspace.get_network(uid) # get arch based on id
        #print(network)
        network.to(device)        

        jacobs_batch = get_batch_jacobian(network, x)
        jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().tolist()

        try:
            s = eval_score_perclass(jacobs_batch, target)

        except Exception as e:
            print(e)
            s = np.nan

        init_population.append((uid,s))
    total_time_cost += (start-time.time()) # add time for all zero-cost proxy processings
    
    # remove C - (C-P)
    init_population = sorted(init_population, key=itemgetter(1), reverse=True)
    init_population = init_population[:P] 
    
    population = []
    history = []

    for ind in reversed(init_population):
        #print(ind)
        # simulate train and get results
        acc12, acc, train_time = searchspace.train_and_eval(ind[0],None, acc_type=acc_type)
        #ind_values = (ind[0],acc12,train_time) # uid, acc, time
        ind_values = (ind[0],acc12,train_time) # uid, acc, time
        population.append(ind_values)
        history.append(ind_values)
        total_time_cost += train_time
    
    C_AUX = C #cyles
    while C_AUX >= 0:
        C_AUX -= 1
        sample = []
        #sample with replacement
        if args.sampling == "S": # sample S candidates
            while len(sample) < S: 
                # Inefficient, but written this way for clarity. In the case of neural
                # nets, the efficiency of this line is irrelevant because training neural
                # nets is the rate-determining step.
                candidate = random.choice(list(population))
                sample.append(candidate)
        elif args.sampling == "highest": # get the highest indv only
            sample.append(max(population, key=itemgetter(1))) #min value
        elif args.sampling == "lowest": # get the lowest indv only
            sample.append(min(population, key=itemgetter(1))) #min value

        parent = max(sample, key=lambda i: i[1]) #get highest acc

        start = time.time()
        generation = []
        P_AUX = P #population size
        #generate generation
        while len(generation) <= P_AUX:
            new_ind = searchspace.mutate_arch(parent[0]) # mutate parent
            new_ind = searchspace[new_ind] # for nasbench101 hash
            network = searchspace.get_network(new_ind).to(device) # get arch based on id

            jacobs_batch = get_batch_jacobian(network, x)
            jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().tolist()

            try:
                s = eval_score_perclass(jacobs_batch, target)
            except Exception as e:
                print(e)
                s = np.nan

            generation.append((new_ind,s))

        total_time_cost += (start-time.time())
        
        chosen_ind = max(generation, key=lambda i: i[1]) #get highest score
        
        acc12, acc, train_time = searchspace.train_and_eval(chosen_ind[0],None, acc_type=acc_type)
        #ind_values = (chosen_ind[0],acc12,train_time)
        ind_values = (chosen_ind[0],acc12,train_time) # uid, acc, time
        population.append(ind_values) # add ind to current pop
        history.append(ind_values) # add ind to history
        total_time_cost += train_time
        
        if args.regularize == 'oldest':
            indv = population[0] #oldest 
        elif args.regularize == 'lowest': # remove lowest scoring 
            indv = min(population, key=itemgetter(1)) #min value
        elif args.regularize == 'highest': # remove highest scoring 
            indv = max(population, key=itemgetter(1)) #min value
        population.pop(population.index(indv))

    times.append(total_time_cost)
    histories.append(history)
    
    top_scoring_arch = max(history, key=lambda i: i[1]) #i[0,1,2] = idx, acc12, time
    best_arch.append(searchspace.get_final_accuracy(top_scoring_arch[0], acc_type, False))
    if not args.dataset == 'cifar10' or args.trainval:
        best_arch_val_acc.append(searchspace.get_final_accuracy(top_scoring_arch[0], val_acc_type, args.trainval))

print(best_arch)

print(histories)
print(f'Mean search time:{np.mean(times):.2f}+/-{np.std(times):.2f}')
if len(best_arch_val_acc) > 0:
    print(f'Mean val  acc:{np.mean(best_arch_val_acc):2.2f}+/-{np.std(best_arch_val_acc):2.2f}')
print(f'Mean test acc:{np.mean(best_arch):2.2f}+/-{np.std(best_arch):2.2f}')
