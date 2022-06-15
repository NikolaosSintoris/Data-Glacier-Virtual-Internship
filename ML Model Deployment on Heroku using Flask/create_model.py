import pandas as pd
import numpy as np
import os
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pickle # pickle used for serializing and de-serializing a Python object structure

file_name = "pima-indians-diabetes.csv"
header_names = ["times_pregnant", "Plasma_glucose_concentration", "Diastolic_blood_pressure", "Triceps_skinfold_thickness", "2h_serum_insulin", "Body_mass_index", "Diabetes_pedigree_function", "Age", "Class"]
data_frame = pd.read_csv(file_name, names = header_names)

# Delete the rows that have 0 in the Plasma_glucose_concentration column
data_frame = data_frame[data_frame.Plasma_glucose_concentration != 0]

# Delete the rows that have 0 in the Body_mass_index column
data_frame = data_frame[data_frame.Body_mass_index != 0]

attributes_names = ["Diastolic_blood_pressure", "Triceps_skinfold_thickness", "2h_serum_insulin"]
for attribute_name in attributes_names:
    mean_value = data_frame[attribute_name].mean()
    data_frame[attribute_name].replace(0, mean_value, inplace = True)

data = data_frame[["times_pregnant", "Plasma_glucose_concentration", "Diastolic_blood_pressure", "Triceps_skinfold_thickness", "2h_serum_insulin", "Body_mass_index", "Diabetes_pedigree_function", "Age"]].to_numpy()
actual_labels = data_frame["Class"].to_numpy()

kf_object = KFold(n_splits=10)
svm_model = svm.SVC()  # create SVM model

accuracy_list = []
f1_score_list = []
for current_train_indices, current_test_indices in kf_object.split(data):
    current_training_dataset = data[current_train_indices[0]:current_train_indices[-1]]
    current_training_actual_labels = actual_labels[current_train_indices[0]:current_train_indices[-1]]

    current_test_dataset = data[current_test_indices[0]:current_test_indices[-1]]
    current_test_actual_labels = actual_labels[current_test_indices[0]:current_test_indices[-1]]

    svm_model.fit(current_training_dataset, current_training_actual_labels)  # Train model with current training dataset
    current_test_predicted_labels = svm_model.predict(current_test_dataset)  # Test model

    current_accuracy = accuracy_score(current_test_actual_labels, current_test_predicted_labels)
    current_f1_score = f1_score(current_test_actual_labels, current_test_predicted_labels)

    accuracy_list.append(current_accuracy)
    f1_score_list.append(current_f1_score)

accuracy_np = np.array(accuracy_list)
print("Mean Accuracy: ", np.mean(accuracy_np))

f1_score_np = np.array(f1_score_list)
print("Mean F1-score: ", np.mean(f1_score_np))

svm_model.fit(data, actual_labels) # Train model with all the data.



# save the model
read_fd = open("model.pkl","wb") # open the file for writing
pickle.dump(svm_model, read_fd) # dumps an object to a file object
read_fd.close() # here we close the fileObject