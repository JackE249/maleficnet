import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.densenet import DenseNet
import os

def save_data(model_name, payload_name, data):
    if not os.path.isdir("./digit_anlysis"):
        os.mkdir("digit_anlysis")
    
    run_num = 0
    while(os.path.isfile(f"./digit_analysis/{model_name}_{payload_name}_{run_num}")):
        run_num += 1

    data.to_csv(f"./digit_analysis/{model_name}_{payload_name}_{run_num}")
    
    

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
    print(cumulative_dist)
    

    

if __name__ == '__main__':
    pass
    

