import pandas as pd
import os
import os.path
results_path = './results'
paths = [f for f in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, f))]
dfs = []
for path in paths:
    df = pd.DataFrame.from_csv(os.path.join(results_path, path))
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(df)
