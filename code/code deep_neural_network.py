
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam




##########"
n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

###########
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])


###########
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr=0.01),  'binary_crossentropy', metrics=['accuracy'])

###########
h=model.fit(x=X, y=y, verbose=1, batch_size=10, epochs=100, shuffle='true')

###########
plt.plot(h.history['acc'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title('accuracy')


############


plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend(['loss'])
plt.title('loss')


############

def plot_descision_limite(X, y, model):# permet de donner une indication par coleurs les zones de chaque classe
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25) # prendre 50 points linéarement équidistant sur l'axe x horizantale
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25) # prendre 50 points linéarement équidistant sur l'axe y verticale
    #print(y_span)
    xx, yy=np.meshgrid(x_span, y_span) 
    # xx est un array de deux dimensions (50, 50) dont tous les lignes sont les memes (x_span)
    # yy est un array de deux dimensions (50, 50) dont tous les colones sont les memes (y_span)
    #print (xx)
    #print (yy)
    xx_, yy_=xx.ravel(), yy.ravel() # conversion en une seule dimension
    #print(yy_)
    #print(xx_)
    grid = np.c_[xx_, yy_] # concatination de deux array chaque 50 elements de xx_ correspond à une valeur de yy-
    #print(grid)
    
    #le but de tous ces fonctions est de préparer un matrice qui contient plusieurs combinaison possibles entre l'axe x et l'axe y 
    #afin de tester les predictions de réseau de neurones 
    
    pred=model.predict(grid) #prediction
    A=pred.reshape(xx.shape) # Redimensionnement en un array de deux dimension 
    #print(A)
    plt.contourf(xx, yy, A) # tracage des contours selon les valeurs de prediction 


################

plot_descision_limite(X, y, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
x1=0.0
y1=0.0
point=np.array([[x1, y1]])
predicition=model.predict(point)
plt.plot([x1], [y1], marker='o', markersize=10, color="red" )
print("predicition", predicition)

