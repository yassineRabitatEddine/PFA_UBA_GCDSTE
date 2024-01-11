import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sb

training_traffic = pd.read_csv("net_trafic.csv")
X=training_traffic
y = training_traffic['Destination']
z=training_traffic['Protocol']

#On convertit les adresses IP en des valeurs numériques
le = LabelEncoder()
X['Destination'] = le.fit_transform(X['Destination'])
y=le.fit_transform(y)

#On convertit les noms des protocoles en des valeurs numériques
le = LabelEncoder()
X['Protocol'] = le.fit_transform(X['Protocol'])
z=le.fit_transform(z)
print("Entraînement du modèle :\n")
print("Données du trafic d'entraînement après conversion en des valeurs numériques (5 premières lignes) : \n")
print(X.head())
print("#################################################\n")

#Normalisation des données
cls = X.columns
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=[cls])
print("Données du trafic d'entraînement après normalisation (5 premières lignes) : \n")
print(X.head())
print("#################################################\n")

#Trouvons le nombre optimal des clusters. Etant donné que le trafic capturé pour le training du modèle ne concerne que deux utilisateurs, nous pourrons
#prouver par la méthode Elbow utilisée ci-après que le nombre de clusters optimal est 2 
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}
#On définit l'erreur par laquelle on peut associer un groupe de données à un centroid (donc un cluster), généralement c'est la somme du carré des distances euclidiennes de chaque point par 
#rapport au centroid choisi 
erreur = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    erreur.append(kmeans.inertia_)

#On trace le résultat avec matplotlib(Fermer la fenêtre du plot une fois le nombre de cluster est consulté pour continuer l'exécution)
plt.plot(range(1, 11), erreur)
plt.xticks(range(1, 11))
plt.xlabel("Nombre de clusters")
plt.ylabel("Erreur calculée")
plt.show()

#On commence le clustering (Training de notre modèle)
kmeans = KMeans(init="random", n_clusters=2, n_init=10, random_state=1)
kmeans.fit(X)
training_traffic['cluster'] = kmeans.labels_
print("Données d'entraînement affectées aux clusters :\n")
print(training_traffic)
print("#################################################\n")
current_labels = [0,1]
desired_labels = ['Utilisateur 1', 'Utilisateur 2']
map_dict = dict(zip(current_labels, desired_labels))

#On trace les clusters à l'aide de matplotlib(fermer la fenêtre du plot pour continuer l'exécution)
sb.scatterplot(data=training_traffic,x="Destination", y="Protocol", hue=training_traffic['cluster'].map(map_dict))
plt.title('Trafic des deux utilisateurs chacun selon un cluster')
plt.show()

#Finalement on teste notre algorithme avec un trafic de test, on entame les mêmes phases de conversion et de normalisation que celles du trafic 
#d'entraîenement
test_trafic = pd.read_csv("net_trafic_de_test.csv")
A=test_trafic
b= test_trafic['Destination']
c=test_trafic['Protocol']

#Convertir les adresses IP en des valeurs numériques
le = LabelEncoder()
A['Destination'] = le.fit_transform(A['Destination'])
b=le.fit_transform(b)

#Convertir les protocoles en des valeurs numériques
le = LabelEncoder()
A['Protocol'] = le.fit_transform(A['Protocol'])
c=le.fit_transform(z)
print("Test du modèle :\n")
print("Données du trafic de test après conversion en des valeurs numériques (5 premières lignes) : \n")
print(A.head())
print("#################################################\n")

#Normalisation des données
cls = A.columns
scaler = MinMaxScaler()
A= scaler.fit_transform(A)
A= pd.DataFrame(A, columns=[cls])
print("Données du trafic de test après normalisation (5 premières lignes) : \n")
print(A.head())
print("#################################################\n")

#Prédiction
kmeans.predict(A)
test_trafic['cluster'] = kmeans.labels_
print("Données de test affectées aux clusters :\n")
print(A)
print("#################################################\n")

#On trace les données de test affectées aux clusters à l'aide de matplotlib
current_labels = [0,1]
desired_labels = ['Utilisateur 1', 'Utilisateur 2']
map_dict = dict(zip(current_labels, desired_labels))
sb.scatterplot(data=test_trafic,x="Destination", y="Protocol", hue=test_trafic['cluster'].map(map_dict))
plt.title('Le trafic de test affecté aux clusters')
plt.show()