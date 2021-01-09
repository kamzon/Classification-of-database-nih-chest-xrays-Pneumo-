# Classification-of-database-nih-chest-xrays-Pneumo-


Open In Colab
Classification de la base doonées chest-xray-pneumonia
Introduction:¶
Le projet consiste à diagnostiquer la pneumonie à partir d'images XRay des poumons d'une personne à l'aide d'un réseau neuronal convolutif auto-posé. Les images étaient de taille supérieure à 1000 pixels par dimension et l'ensemble de données total était étiqueté grand et avait un espace de 1,2 Go. notre travail comprend un réseau de neurones auto-étendu qui a été réglé à plusieurs reprises pour l'un des meilleurs hyperparamètres et utilisé une variété de fonctions d'utilité de Keras comme les rappels pour la réduction du taux d'apprentissage et les points de contrôle. Nous allons d'abord parcourir l'ensemble de données de formation. Nous allons faire une analyse à ce sujet, examiner certains des échantillons, vérifier le nombre d'échantillons pour chaque classe, etc.

Nous avons trois répertoires ( train , test, validation) et chaques répertoires contient deux sous-répertoires ci dessous :

NORMAL: Ce sont les échantillons qui décrivent le cas normal (pas de pneumonie).
PNEUMONIE: Ce répertoire contient les échantillons qui sont les cas de pneumonie.
Le objectif de ce projet est :
la réalisation d’un modèle CNN pour la classification de la base de données chest-xray-pneumonia avec le framework Keras et représentation de loss et accuracy sous forme des diagrammes.

les outils utulisés :
Model CNN
Un CNN est simplement un empilement de plusieurs couches de convolution, pooling, correction ReLU et fully-connected. Chaque image reçue en entrée va donc être filtrée, réduite et corrigée plusieurs fois, pour finalement former un vecteur. Dans le problème de classification, ce vecteur contient les probabilités d'appartenance aux classes. Tous les réseaux de neurones convolutifs doivent commencer par une couche de convolution et finir par une couche fully-connected. Les couches
intermédiaires peuvent s'empiler de différentes manières,à condition que la sortie d'une couche ait la même structure que l'entréede la suivante. Par exemple, une couche fully-connected, qui renvoie toujours un vecteur, ne peut pas être placée avant une couche de pooling, puisque cette dernière doit recevoir une matrice 3D.

Tansorflow
Les API de haut niveau de TensorFlow sont basées sur la norme API Keras pour définir et former des réseaux de neurones. Keras permet un prototypage rapide, une recherche et une production de pointe, le tout avec des API conviviales.

Keras
Keras est une bibliothèque de réseaux neuronaux de haut niveau qui s'exécute au sommet de TensorFlow, CNTK et Theano. L'utilisation de Keras dans le deep learning permet un prototypage facile et rapide ainsi qu'une exécution transparente sur CPU et GPU. Ce framework est écrit en code Python qui est facile à déboguer et permet une facilité d'extensibilité.

Réalisation
Voici les étapes nécessaires que nous avons suivi pour classifier la base de données chest-xray-pneumonia à base d’un modèle CNN:

1. Téléchargement de la base de donnée a partir de kaggle:

2. extraction de la base donnée:

3. importation des bibliothèques pour le praitraittement de la base de donnée:

4. Définir un echatillant pour l'entrainement du model apartir de la base de données (x_train,Y_train):
 
5. Défenir un echatillant pour la validation du medel apartir de la base de données(X_validation, Y_validation):

6. Défenir un echatillant pour tester medel apartir de la base de données (X_test, Y_test):

7. Analyse de la base de donnée:

8. Definition des models:
