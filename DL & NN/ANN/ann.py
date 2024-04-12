import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input  # Import Input layer

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

# INITIALIZING ANN
classifier = Sequential()

# Add Input layer as the first layer
classifier.add(Input(shape=(11,)))

# Add hidden layers
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dense(units=6, activation='relu'))

# Add output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train our NN
classifier.fit(x_train, y_train, batch_size=10,epochs=100)

# Predict Test Data
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
print(cm)
