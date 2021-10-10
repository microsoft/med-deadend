from utils import make_train_val_test_split

# 80% training data + 5% validation data + 15% test data
train_fraction = 0.8
validation_fraction = 0.05

data_file = r"./data/sepsis_mimiciii/sepsis_final_data_K1.csv"

# will save the split data in the same folder as data_file
print("Processing ...")
make_train_val_test_split(filename=data_file, train_frac=train_fraction, val_frac=validation_fraction, make_test=True)
print("Done.")
