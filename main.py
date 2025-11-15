import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from train_model import train_model
from preprocess_data import preprocess_data

#modif etudiant B
from sklearn.metrics import confusion_matrix, classification_report

print("Matrice de confusion :")
print(confusion_matrix(test_y, y_pred))

print("\nRapport de classification :")
print(classification_report(test_y, y_pred))
#fin de modif

# Chargement des données
iris = pd.read_csv("InputData/Iris.csv")
test_size = 0.3

# Prétraitement
train, test = preprocess_data(iris, test_size)

# Séparation des features et des cibles
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y = train.Species
test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y = test.Species

# Initialisation du modèle
model = svm.SVC()

# Entraînement et prédiction
prediction = train_model(train_X, train_y, test_X, model)

# Affichage de la précision
accuracy = accuracy_score(test_y, prediction)
print(f"Accuracy du modèle SVM : {accuracy:.2f}")

# Courbe d’apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    model, train_X, train_y, cv=5, scoring='accuracy',
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
)

train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Score entraînement')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Score validation')
plt.title('Courbe d’apprentissage du modèle SVM')
plt.xlabel('Taille de l’échantillon d’entraînement')
plt.ylabel('Précision')
plt.legend(loc='best')
plt.grid()
plt.show()
