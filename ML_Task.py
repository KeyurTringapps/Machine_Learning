
# generate related variables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1 - Read the data from data source 
df = pd.read_csv('./housing.csv')
#print(df.columns)
print(df.isna().sum())
#print(df.info)
#print(df.shape)
# Step 2 - Clean the data / prepare the data for ML operation
df['ocean_proximity'] = df['ocean_proximity'].replace(['NEAR BAY', 'INLAND', '<1H OCEAN', 'ISLAND', 'NEAR OCEAN'], [1, 2, 3, 4, 5])

total_bedrooms_median = df['total_bedrooms'].median()

df['total_bedrooms'] = df['total_bedrooms'].fillna(total_bedrooms_median)
#print(df.info)
print(df.isna().sum())
print("+++++++++++++++++++")

# Step 3 - Create the model
x_axis = df.drop('median_house_value', axis=1)
y_axis = df['median_house_value']
print("++++++++y_axis+++++++++++")
print(y_axis)
print("+++++++++y_axis++++++++++")
x_axis_train, x_axis_test, y_axis_train, y_axis_test = train_test_split(x_axis, y_axis, train_size=0.8 ,test_size=0.2, random_state=90)


# Step 4 - Perform the operation (predict the value)

def random_forest():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=5)
    model.fit(x_axis_train, y_axis_train)
    return  model


# Step 5 - Model Evaluation 
def evaluate_mode(model, model_name):
    y_predictions = model.predict(x_axis_test)

    print(f'======= {model_name} ========')
    from sklearn.metrics import  confusion_matrix, accuracy_score
    cm = confusion_matrix(y_axis_test, y_predictions)

    print(f'======= accuracy_score ========')
    print(f'accuracy = {accuracy_score(y_axis_test, y_predictions) * 100 : 0.2f}%')



# My Methods Of Models
def logistic_regression():
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV()
    model.fit(x_axis_train, y_axis_train)
    return model

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_axis_train, y_axis_train)
    return  model

def svm(c):
    from sklearn.svm import SVC
    model = SVC(C=c, gamma=0.001)
    model.fit(x_axis_train, y_axis_train)
    return  model

def decission_tree():
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(x_axis_train, y_axis_train)
    return  model


# Step 6 - Data Visulization of result 
def plot_graph(model):
    plt.plot(model, color='green', label="ML")

    plt.legend()

    plt.title("My Machine Learning")
    plt.xlabel("")
    plt.ylabel("")

    plt.show()

# Calling Function

#model_rf = random_forest()
#evaluate_mode(model_rf, 'Random Forest')

#model_lr = logistic_regression()
#evaluate_mode(model_lr, 'Logistic Regression')

#model_knn = knn()
#evaluate_mode(model_knn, 'KNN')

#model_svm = svm(c=2.0)
#evaluate_mode(model_svm, 'SVN')

model_decission_tree = decission_tree()
evaluate_mode(model_decission_tree, 'Decission Tree')

