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
for result_file in files:
    df = pd.read_csv(result_file)
    df = df[(df["up"] > 1.6)]
    print(df['target_price_change1'].values)
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print(result_file)
    mean_50 = df['target_price_change1'].mean()
    if pd.isnull(mean_50):
        continue
    base = base * (1+ mean_50 * 0.01 )
    #print('base: ' + str(base))
    #print(mean_50)
    #mean = df.sort_values(by=['2-20170913-cnn_three_mean'], ascending=False)['target_price_change1'].mean()
    #print(mean_50, mean, mean_50 - mean)
