import csv
print("\nLoading the dataset from file ...")

def load_dataset(file_path):
    dataset = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                dataset.append({'center':line[0], 'left':line[1], 'right':line[2], 'steering':float(line[3]),
                            'throttle':float(line[4]), 'brake':float(line[5]), 'speed':float(line[6])})
            except:
                continue # some images throw error during loading
    return dataset

dataset = load_dataset('C:\\Users\\kiit1\\Documents\\steering angle prediction\\dataset_coldivision\\data\\driving_log.csv')
print("Loaded {} samples from file {}".format(len(dataset),'C:\\Users\\kiit1\\Documents\\steering angle prediction\\dataset_coldivision\\data\\driving_log.csv'))

from random import shuffle
from sklearn.model_selection import train_test_split

print("Partioning the dataset:")

shuffle(dataset)

#partitioning data into 80% training, 19% validation and 1% testing

X_train,X_validation=train_test_split(dataset,test_size=0.2)
X_validation,X_test=train_test_split(X_validation,test_size=0.05)

print("X_train has {} elements.".format(len(X_train)))
print("X_validation has {} elements.".format(len(X_validation)))
print("X_test has {} elements.".format(len(X_test)))
print("Partitioning the dataset complete.")
