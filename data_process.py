import pandas as pd
import os
from utils import make_train_val_test_split


# 80% training data + 5% validation data + 15% test data
train_fraction = 0.8
validation_fraction = 0.05

data_file = r"./data/sepsis_mimiciii/sepsis_final_data_K1.csv"


# remove extra columns
data = pd.read_csv(data_file)
save = False
for item in ['o:input_total', 'o:input_4hourly', 'o:max_dose_vaso']:
    if item in data.columns:
        data.drop(labels=[item], axis=1, inplace=True)
        save = True
if save:
    os.rename(data_file, data_file[:-4] + '_fullcolumns.csv')
    data.to_csv(os.path.join(data_file))  # overwrite with new data_file


# will save the split data in the same folder as data_file
print("Processing ...")
make_train_val_test_split(filename=data_file, train_frac=train_fraction, val_frac=validation_fraction, make_test=True)
print("Done.")
