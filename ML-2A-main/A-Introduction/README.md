# ML-2A -- Introduction et premier algorithme *à la main*

## Tutoriels


Ce tutoriel est basé sur des notebook jupyter, il requiert:
* une distribution scientifique de python, la plus répendue étant anaconda [lien](https://www.anaconda.com)
* [mac/linux] pour travailler sur les canevas proposés, il suffit ensuite de taper:
  jupyter-notebook <fichier.ipynb> 
* [windows] naviguer dans anaconda pour charger le fichier
* [online] en cas de problème, les liens ci-dessous donne accès aux mêmes TP en ligne via google colab
  * il faut un compte gmail et une connexion internet pour en profiter
* [option avancée] éventuellement un éditeur avancé capable de gérer les notebooks: VS Code [lien](https://code.visualstudio.com)
VS Code est très bien fait: à l'ouverture des fichiers, en fonction des extensions, il propose de télécharger des plugins pour gérer les spécificités desdits fichiers.


**L'introduction vise à mettre tout le monde à niveau sur:**
- python
- numpy, matplotlib (et un peu de pandas)

L'introduction est particulièrement dense... Mais la plupart des informations transmises sont optionnelles. L'enjeu est de survoler, de retenir l'endroi où se trouvent les informations pour les récupérer ultérieurement si besoin.

Les notebooks sont numérotés de 1 à 7: le premier doit déjà être acquis, les 6 et 7 sont assez facultatifs... On se concentre donc sur les 2, 3, 4, 5

## Naïve Bayes: premier modèle à la main

Une sorte de tutoriel orienté données. L'objectif est de consolider la pratique à travers un cas d'usage ludique: classer des images de chiffres manuscrits provenant d'enveloppes postales (des codes postaux décomposés).

- Choisir une modélisation statistique
- Calculer les paramètres des distributions au sens du max de vraisemblance
- Prendre une décision et mesurer les performances

<img src="/A-Introduction/3-Naive-Bayes/notebooks/ressources/usps.png" width=200px>


## Gradient

Dernier exercice consacré à numpy: nous allons apprendre un régresseur par descente de gradient afin de réviser:
1. numpy
2. la problématique de la régression
3. l'algorithme de la descente de gradient [qui est à la base des réseaux de neurones!]

De nouveau, nous consolidons les bases de numpy/matplotlib... Tout en poursuivant le panorama du ML vers les problèmes de régression et de gradient. Ce dernier point est critique: le gradient est un outil central dans tous les réseaux de neurones et il est important d'avoir une représentation mentale de l'impact des différents réglages possibles.

<img src="/A-Introduction/3-Gradient/notebooks/fig/animation.gif" width=600px>