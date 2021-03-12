import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv(
    '/media/lepoeme20/cloud/projects/20_daewoo/convlstm/result/result_source__split_0_seed_0/result_source__target_brave.csv' 
    )

# plt.figure(figsize=(12, 5))
# plt.scatter(np.arange(len(results)), results['pred'], c='red', s=1, label='Pred')
# plt.scatter(np.arange(len(results)), results['true'], c='blue', s=1, label='True')
# plt.legend()

# plt.savefig('./result/result_non_datetime.jpg')
# plt.close()

all_data = pd.read_csv(
    '/media/lepoeme20/cloud/projects/20_daewoo/preprocessing/brave_data_label.csv'
)

test_set = all_data[all_data['iid_phase']=='test']
time = test_set['time'].values
results['time'] = time

plt.figure(figsize=(16, 5))
plt.scatter(pd.to_datetime(results['time']), results['pred'], c='red', s=1, label='Pred')
plt.scatter(pd.to_datetime(results['time']), results['true'], c='blue', s=1, label='True')
plt.legend()

plt.savefig('./result/result.jpg')
plt.close()