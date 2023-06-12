import numpy as np
import pandas as pd
import os

from models import DenseNet
import torch

import multiprocessing as mp

def save_data(model_name, payload_name, data):
    if not os.path.isdir("./digit_analysis"):
        os.mkdir("digit_analysis")
    
    run_num = 0
    while(os.path.isfile(f"./digit_analysis/{model_name}_{payload_name}_{run_num}")):
        run_num += 1

    data.to_csv(f"./digit_analysis/{model_name}_{payload_name}_{run_num}.csv")
    
    

def get_digit_distribution(model) -> pd.Series:
    def leading_digit(f: float):
        if f < 1:
            return 0
        while (f >= 10):
            f = f // 10
        return int(f)
    fast_leading_digit = np.vectorize(leading_digit)

    state_dict = model.state_dict()
    relevant_labels = [n for n in state_dict.keys() if "weight" in str(n)][:-1]
    cumulative_dist = pd.Series(data=[0] * 10)
    for label in relevant_labels:
        weights: np.ndarray = state_dict[label].detach().cpu().numpy().flatten()
        weights *= (10 ** 10)
        weights = np.abs(weights)
        vals = pd.Series(weights).apply(fast_leading_digit).value_counts()
        for i in range(10):
            if i in vals.index:
                cumulative_dist[i] += vals[i]
    return cumulative_dist

def worker(model_name, model_type = "densenet_169") -> pd.Series:
    model_path = f"./checkpoints/{model_name}"
    if model_type == "densenet_169":
        model = DenseNet(input_shape=32,
                        num_classes=10,
                        only_pretrained=False,
                        model_size=169)
    model.load_state_dict(torch.load(model_path))
    return get_digit_distribution(model)

    

if __name__ == '__main__':
    SKIP_CLEAN = True
    if not SKIP_CLEAN:
        clean_models = []
        for model_name in os.listdir("./checkpoints"):
            if model_name[:23] != "densenet_cifar10_clean_":
                continue
            clean_models.append(model_name)
        
        with mp.Pool() as p:
            distributions = p.map(worker, clean_models)
            data = dict(zip(clean_models, distributions))
            pd.DataFrame(data).to_csv("./digit_analysis/clean_dist.csv", index=False)
    
    SKIP_EPOCHS = False
    if not SKIP_EPOCHS:
        epoch_models = []
        for model_name in os.listdir("./checkpoints"):
            if model_name[:23] != "densenet_cifar10_epoch_":
                continue
            epoch_models.append(model_name)
        
        with mp.Pool() as p:
            distributions = p.map(worker, epoch_models)
            data = dict(zip(epoch_models, distributions))
            pd.DataFrame(data).to_csv("./digit_analysis/epoch_dist.csv", index=False)


    

