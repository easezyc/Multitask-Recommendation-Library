import pandas as pd
import numpy as np
import pickle
import zipfile
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='NL', choices=['NL', 'ES', 'FR', 'US'])
args = parser.parse_args()

dataset_name = args.dataset_name
data_path = './data/'

zip_file = zipfile.ZipFile(data_path + 'aliexpress_{}_datasets.zip'.format(dataset_name))
zip_list = zip_file.namelist()
for f in zip_list:
    zip_file.extract(f, data_path)
zip_file.close()
os.rename(data_path + 'aliexpress_{}_datasets'.format(dataset_name), data_path + 'AliExpress_{}'.format(dataset_name))

for item in ['_item_train.zip', '_item_test.zip', '_user_train.zip', '_user_test.zip']:
    zip_file = zipfile.ZipFile(data_path + 'AliExpress_{}/'.format(dataset_name) + '{}'.format(dataset_name.lower()) + item)
    zip_list = zip_file.namelist()
    for f in zip_list:
        zip_file.extract(f, data_path + 'AliExpress_{}/'.format(dataset_name))
    zip_file.close()


df_train_item = pd.read_csv('./data/AliExpress_{}/{}_item_train.csv'.format(dataset_name, dataset_name.lower()), header=None, names=['item_{}'.format(i) for i in range(1, 50)])
df_test_item = pd.read_csv('./data/AliExpress_{}/{}_item_test.csv'.format(dataset_name, dataset_name.lower()), header=None, names=['item_{}'.format(i) for i in range(1, 50)])
df_train_user = pd.read_csv('./data/AliExpress_{}/{}_user_train.csv'.format(dataset_name, dataset_name.lower()), header=None, names=['user_{}'.format(i) for i in range(1, 34)])
df_test_user = pd.read_csv('./data/AliExpress_{}/{}_user_test.csv'.format(dataset_name, dataset_name.lower()), header=None, names=['user_{}'.format(i) for i in range(1, 34)])

df_test = pd.merge(df_test_user, df_test_item, left_on='user_1', right_on='item_1', how='inner')
df_train = pd.merge(df_train_user, df_train_item, left_on='user_1', right_on='item_1', how='inner')

num_train = df_train.shape[0]
num_test = df_test.shape[0]
print('train: {}; test: {}'.format(num_train, num_test))

df_data = pd.concat([df_train, df_test]).reset_index(drop=True)

category_cols = ['user_{}'.format(i) for i in range(1, 9)] + ['user_{}'.format(i) for i in range(32, 34)] + ['item_{}'.format(i) for i in [19, 34, 35, 36, 40, 41, 42]]
item_index = [i for i in range(2, 49)]
for i in [19, 34, 35, 36, 40, 41, 42]:
    item_index.remove(i)
numerical_cols = ['user_{}'.format(i) for i in range(9, 32)] + ['item_{}'.format(i) for i in item_index]
label_col = 'item_49'

df_category_data = df_data[category_cols]
df_numerical_data = df_data[numerical_cols]
df_label_data = df_data[label_col]

new_category_cols = ['search_id'] + ['categorical_{}'.format(i) for i in range(1, len(category_cols))]
df_category_data.columns = new_category_cols
new_numerical_cols = ['numerical_{}'.format(i) for i in range(1, len(numerical_cols) + 1)]
df_numerical_data.columns = new_numerical_cols

label_list = []
for label in df_label_data.values:
    if label == 0:
        label_list.append([0, 0])
    elif label == 1:
        label_list.append([1, 0])
    elif label == 2:
        label_list.append([1, 1])
df_label = pd.DataFrame(np.array(label_list), columns=['click', 'conversion'])

df_final_data = pd.concat([df_category_data, df_numerical_data, df_label], axis=1)
print('categorical col num: {}; numerical col num: {}'.format(len(category_cols) - 1, len(numerical_cols)))

df_final_train = df_final_data.iloc[:num_train]
df_final_test = df_final_data.iloc[num_train:]

df_final_train.to_csv('./data/AliExpress_{}/train.csv'.format(dataset_name), index=0)
df_final_test.to_csv('./data/AliExpress_{}/test.csv'.format(dataset_name), index=0)