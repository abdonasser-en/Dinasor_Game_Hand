# Projet : Dinosaur_game_with_hand

## Introduction

Ce projet a pour objectif de mettre tous les bagages requis tout au long de L'UV **3Dtchnologies** . En effet,Nous simulons un jeux, dans notre cas, c'est le jeu T-rex dinosaur de google chrome qu'on l'a codé avec la bibliothèque pygame dans python, basé sur la détection de la main. Dans ce fichier **_Readme_**, nous allons expliquer en détail l'utilisation du jeu. 

### Installation

Les bibliothéques requises:

```sh
pip3 install gypame
pip3 install mediapipe
pip3 install opencv-python
pip3 install pandas
pip3 install numpy
```

### Scripts Python

- **post_recog.py**

    > Dans ce fichier, il y a le jeu, et l'algorithme de detéction par la main en utilisant la caméra.

- **collect_data.py**

  > Dans ce fichier, on collecte les données  pour la main ouverte et fermée en utilsant la même algorithme du fichier pos_recog.
  
- **ClassifierHandModel.ipynb**

   > Dans ce fichier , on entraine un modéle de classification (K neighboors) pour la positionnement de notre main en face de la caméra.


### Dépendances

- ****knn.pickle****

   > notre modéle entrainé en format pickel 
- **class0_data.csv**
   > Les données collectées pour la main ouverte.
- **__class1_data.csv__**
   > Les données collectées pour la main fermée.
- **\img**
   > Un dossier qui contient des images  du jeu T-rex dinasour.

### Utilisation 

On exécute le fichier python post_recog qui lancera deux fenêtres, la première est la camera qui détectera les 21 keypoints de notre main et l'autre fenêtre est le jeu.

On lance le jeu et grâce à notre main détectée par la camera, on donne l'ordre au dinosaure de sauter ou pas, le dinosaure doit sauter au-dessus du cactus pour ne pas perdre.
 
