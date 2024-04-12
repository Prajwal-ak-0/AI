import numpy as np
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# HANDLING CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x1 = LabelEncoder()
x[:, 1] = labelencoder_x1.fit_transform(x[:, 1])

labelencoder_x2 = LabelEncoder()
x[:, 2] = labelencoder_x2.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categories='auto', drop='first')

x_encoded = onehotencoder.fit_transform(x[:, [1, 2]])

x_encoded = x_encoded.toarray()
x = np.concatenate((x[:, [0]], x_encoded, x[:, 3:]), axis=1)

# SPLITTING OF TRAIN AND TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# IMPORTING LIBRARIES FOR INTIALIZING ANN  
from keras.models import Sequential
from keras.layers import Dense, Input

# IMPORTING "DROPOUT" LIBRARY IN CASE OF OVERFITTING SCENARIO
from keras.layers import Dropout

classifier = Sequential()

# Add Input layer as the first layer
classifier.add(Input(shape=(11,)))
# classifier.add(Dropout(rate = 0.1))

# Add hidden layers
classifier.add(Dense(units=6, activation='relu'))
# classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units=6, activation='relu'))
# classifier.add(Dropout(rate = 0.1))

# Add output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train our NN
classifier.fit(x_train, y_train, batch_size=10,epochs=100)

# Predict Test Data
y_pred_test = classifier.predict(x_test)

# Convert probabilities to binary predictions
y_pred_test = (y_pred_test > 0.5)  

# Evaluate model performance using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# New data point to predict

# y_p = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1,40, 3, 0000, 2, 1, 1, 90000]])))
# y_p = (y_p > 0.5)
# print(y_p)

# PLOTTING TEST AND TRAIN SET ACCURACIES GRAPH

# import matplotlib.pyplot as plt
# # Train the model and store the history
# history = classifier.fit(x_train, y_train, batch_size=10, epochs=100, validation_data=(x_test, y_test))

# # Plot training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Plot training and validation accuracy
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


from scikeras.wrappers import KerasClassifier 
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Input(shape=(11,)))
    classifier.add(Dense(units=6, activation='relu'))
    classifier.add(Dense(units=6, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier


classifier = KerasClassifier(model=build_classifier, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=1)
mean = accuracies.mean()
variance = accuracies.std()
print("Mean accuracy:", mean)
print("Variance:", variance)


    
# PARAMATER'S TUNING FOR OPTIMIZING OUR ANN
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', input_shape=(11,)))
    classifier.add(Dense(units=6, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(model=build_classifier)
parameters = {
        'epochs' : [100, 200],
        'optimizer' : ['adam', 'rmsprop']
    }
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10) 
grid_search = grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
 
    
    
    
    
    
    
    
    
    
    
    
    