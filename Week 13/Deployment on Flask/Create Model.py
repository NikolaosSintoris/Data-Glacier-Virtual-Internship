import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pickle # pickle used for serializing and de-serializing a Python object structure

#================================ Load Data ===============================================================================
# Find the current working directory and the datasets directory.
current_working_directory = os.getcwd()
dataset_directory = current_working_directory + "\DataSet"

# Load my csv data to a data frame.
bank_data_df = pd.read_csv(dataset_directory + '\\bank-additional-full.csv', sep = ";")

#=========================================================================================================================

#================================ Missing Values ===============================================================================

# Convert the column 'y' from categorical to numeric because it will help me in the future.
# 1 is 'yes and 0 is 'no'
bank_data_df['y'].replace(['yes', 'no'], [1, 0], inplace = True)

# For the missing values we are going to use 2 techniques.
# When the most frequent word has by far more data points we, are going to replace the missing values with the most frequent one.
# Otherwise we are going to replace the missing values with a random value.
# So, for the job, education and housing attributes we are going to use the second technique and the first technique for the other attributes.
collumn_names_missing_values_list = ["marital", "default", "loan"]

# A function that for every attribute, it replaces every unknown value with the most frequent one.
def replace_missing_values(collumn_name):
    result_a = bank_data_df[collumn_name].value_counts()
    most_frequent_value = result_a.index[0]
    bank_data_df[collumn_name].replace("unknown", most_frequent_value, inplace = True)

for current_collumn_name in collumn_names_missing_values_list:
    replace_missing_values(current_collumn_name)

# Replace the unknown values of attributes job, education and housing with a random values.
def replace_missing_values_with_random_ones(attribute_name):
    attribute_unique_values_list = bank_data_df[attribute_name].unique().tolist() # Create a list that has all the unique values of the attribute
    attribute_unique_values_list.remove("unknown") # Remove the value unknown from the list
    random_value = random.choice(attribute_unique_values_list) # Pick a random value from the list
    print("Random Value: ", random_value)
    bank_data_df[attribute_name].replace("unknown", random_value, inplace = True)

replace_missing_values_with_random_ones("job")
replace_missing_values_with_random_ones("education")
replace_missing_values_with_random_ones("housing")

#=============================================================================================================================

#================================ Model Building ===============================================================================

# Get all categorical columns
categorical_columns = bank_data_df.select_dtypes(['object']).columns
# Convert all categorical columns to numeric
bank_data_df[categorical_columns] = bank_data_df[categorical_columns].apply(lambda x: pd.factorize(x)[0])

data = bank_data_df.drop(columns = ['y'])
targets = bank_data_df['y'] # Actual labels of our data.

k_fold_cv_object = KFold(n_splits = 10) # K fold cross validation with 10 folds.

accuracy_list = []
f1_score_list = []
recall_list = []
precision_list = []

random_forest_model = RandomForestClassifier(random_state=0)  # Create model.
for train_index, test_index in k_fold_cv_object.split(data):
    # Create train and test sets.
    data_train, data_test = data[train_index[0]:train_index[-1]], data[test_index[0]:test_index[-1]]
    targets_train, targets_test = targets[train_index[0]:train_index[-1]], targets[test_index[0]:test_index[-1]]

    # Train and test model.
    random_forest_model.fit(data_train, targets_train)  # Train model.
    predicted_targets = random_forest_model.predict(data_test)  # Test model.

    # Compute accuracy, f1-score, recall and precision.
    accuracy_list.append(accuracy_score(targets_test, predicted_targets))
    f1_score_list.append(f1_score(targets_test, predicted_targets, average='macro'))
    recall_list.append(recall_score(targets_test, predicted_targets, average='macro'))
    precision_list.append(precision_score(targets_test, predicted_targets, average='macro'))

# Convert lists to numpy arrays.
accuracy_np = np.array(accuracy_list)
f1_score_np = np.array(f1_score_list)
recall_np = np.array(recall_list)
precision_np = np.array(precision_list)

# Compute mean value from every metric.
random_forest_accuracy = np.mean(accuracy_np)
random_forest_f1_score = np.mean(f1_score_np)
random_forest_recall = np.mean(recall_np)
random_forest_precision = np.mean(precision_np)

# dictionary with list object in values
results_dict = {
    'Random Forest Classifier': [random_forest_accuracy, random_forest_f1_score, random_forest_recall, random_forest_precision],
}

# creating a Dataframe object from dictionary
results_df = pd.DataFrame(results_dict, index=['Accuracy', 'F1-Score', 'Recall', 'Precision'])

print("Results from Random Forest Classifier Training\n")
print(results_df)

# =============================================================================================================================

# ============================================ Feature Importance =============================================================
final_data = bank_data_df[['duration', 'euribor3m', 'age', 'job', 'campaign', 'pdays', 'education', 'day_of_week', 'nr.employed', 'poutcome']]
final_targets = bank_data_df['y'] # Actual labels of our data.

final_accuracy_list = []
final_f1_score_list = []
final_recall_list = []
final_precision_list = []

final_model = RandomForestClassifier(random_state=0)  # Create model.
for train_index, test_index in k_fold_cv_object.split(final_data):
    # Create train and test sets.
    data_train, data_test = final_data[train_index[0]:train_index[-1]], final_data[test_index[0]:test_index[-1]]
    targets_train, targets_test = final_targets[train_index[0]:train_index[-1]], final_targets[
                                                                                 test_index[0]:test_index[-1]]

    # Train and test model.
    final_model.fit(data_train, targets_train)  # Train model.
    predicted_targets = final_model.predict(data_test)  # Test model.

    # Compute accuracy, f1-score, recall and precision.
    final_accuracy_list.append(accuracy_score(targets_test, predicted_targets))
    final_f1_score_list.append(f1_score(targets_test, predicted_targets, average='macro'))
    final_recall_list.append(recall_score(targets_test, predicted_targets, average='macro'))
    final_precision_list.append(precision_score(targets_test, predicted_targets, average='macro'))

# Convert lists to numpy arrays.
accuracy_np = np.array(final_accuracy_list)
f1_score_np = np.array(final_f1_score_list)
recall_np = np.array(final_recall_list)
precision_np = np.array(final_precision_list)

# Compute mean value from every metric.
final_accuracy = np.mean(accuracy_np)
final_f1_score = np.mean(f1_score_np)
final_recall = np.mean(recall_np)
final_precision = np.mean(precision_np)

# dictionary with list object in values
final_results_dict = {
    'Random Forest Classifier': [final_accuracy, final_f1_score, final_recall, final_precision],
}

# creating a Dataframe object from dictionary
final_results_df = pd.DataFrame(final_results_dict, index=['Accuracy', 'F1-Score', 'Recall', 'Precision'])

print("Final results from training\n")
print(final_results_df)

# Train model with all the data once.
final_model.fit(final_data, final_targets)  # Train model.

# save the model
read_fd = open("model.pkl","wb") # open the file for writing
pickle.dump(final_model, read_fd) # dumps an object to a file object
read_fd.close() # here we close the fileObject
# =============================================================================================================================