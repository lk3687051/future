import pandas as pd
import os
def get_files(path):
    file_list = []
    for root,dirs,files in os.walk(path):
        for filespath in files:
            if 'csv' in filespath:
                file_list.append(os.path.join(root,filespath))
    return file_list
files = get_files('./results')
base = 100
count = 0
radios = []
for result_file in files:
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print(result_file)
    df = pd.read_csv(result_file)
    mean_up = df['up'].mean()
    mean_change = df['target_price_change1'].mean()
    print(mean_up, mean_change)
    df = df[(df["up"] > 1.3)]

    #mean_50 = df.sort_values(by=['2-20170913-cnn_three_mean'], ascending=False).head(1)['target_price_change1'].mean()
    mean_50 = df.sort_values(by=['up'], ascending=False).head(1)['target_price_change1'].mean()

    if pd.isnull(mean_50):
        continue
    count = count + 1
    radios.append(mean_50)
    base = base * (1+ mean_50 * 0.01 - 0.0015)
    print('base: ' + str(base))

import numpy as np
print(np.mean(radios))
print(sorted(radios))
