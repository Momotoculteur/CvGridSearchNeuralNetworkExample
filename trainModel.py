from keras import Sequential
from keras.datasets import mnist
import numpy as np
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# HYPER PARAM
EPOCH = [1]
BATCH_SIZE = [500]
OPTIMIZER = ['rmsprop', 'adam']

# Dictionnaire contenant l'ensemble des valeurs des hyperparams
hyperMatrix = dict(optimizer=OPTIMIZER, epochs=EPOCH, batch_size=BATCH_SIZE)

# On récup le dataset MNIST et on le preprocess
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],28, 28,1)
X_test = X_test.reshape(X_test.shape[0],28, 28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalisation des données : {0-255} => {0-1}
X_train /= 255
X_test /= 255

# One hot encoding des labels
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# Définition du model
def buildModel(optimizer="adam"):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

#On wrap le model dans un KerasClassifier pour permettre de l'utiliser dans les API de ScikitLearn
model = KerasClassifier(build_fn=buildModel, verbose=1)

# parra : , n_jobs=-1

# Grid qui associe un model avec une liste d'hyperparam à tester
grid = GridSearchCV(estimator=model, param_grid=hyperMatrix)

# On lance l'entrainement
history = grid.fit(X_train, Y_train)

# On affiche la combinaison d'hyper param qui nous a donné les meilleurs score
print('Meilleur combinaison de param')
print(history.best_params_)
print("Meilleur score")
print(history.best_score_ )
print("Perf sur l'ensemble des combinaisons")
print(history.cv_results_)




print("Evaluation du meilleur model sur dataset test")


# Pour tester notre meilleur modele sur le dataset de test
bestModel = history.best_estimator_.model
metricsName = bestModel.metrics_names
metricsVal = bestModel.evaluate(X_test, Y_test)
for metric, value in zip(metricsName, metricsVal):
    print(metric, ': ', value)


