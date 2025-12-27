# ✍️ Reconnaissance d'Écriture Manuscrite

Projet de Deep Learning pour reconnaître les chiffres manuscrits (0-9) en utilisant un réseau de neurones convolutif (CNN).

## 📋 Description

Ce projet implémente un **CNN (Convolutional Neural Network)** pour la reconnaissance de chiffres manuscrits sur le célèbre dataset **MNIST**. Le modèle est capable de reconnaître avec haute précision les chiffres écrits à la main.

### 🎯 Objectifs :
- Entraîner un modèle CNN sur 60,000 images de chiffres manuscrits
- Atteindre une précision supérieure à 98%
- Visualiser les prédictions et les probabilités
- Permettre des tests sur images personnalisées

## 🛠️ Technologies utilisées

- **TensorFlow/Keras** : Framework de Deep Learning
- **NumPy** : Manipulation de données
- **Matplotlib** : Visualisations
- **PIL (Pillow)** : Traitement d'images

## 📁 Structure du projet

```
reconnaissance-ecriture-manuscrite/
│
├── src/
│   ├── data_loader.py      # Chargement et préparation du dataset MNIST
│   ├── model.py             # Architecture du CNN
│   ├── train.py             # Script d'entraînement
│   └── predict.py           # Prédictions et visualisations
│
├── models/                  # Modèles entraînés (générés)
├── examples/                # Images de résultats (générées)
├── requirements.txt         # Dépendances Python
└── README.md
```

## 🚀 Installation

1. Cloner le repository :
```bash
git clone https://github.com/ilyes-elhamdi/reconnaissance-ecriture-manuscrite.git
cd reconnaissance-ecriture-manuscrite
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### 1. Entraîner le modèle

**Mode complet** (meilleure performance, ~10 minutes) :
```bash
cd src
python train.py
```

**Mode rapide** (test rapide, ~3 minutes) :
```bash
cd src
python train.py --simple
```

Cela va :
- Télécharger le dataset MNIST automatiquement
- Créer et entraîner le modèle CNN
- Sauvegarder le modèle dans `models/`
- Générer des graphiques d'entraînement

### 2. Tester le modèle

```bash
cd src
python predict.py
```

Cela va :
- Charger le modèle entraîné
- Tester sur 20 échantillons aléatoires
- Évaluer sur tout le dataset de test
- Sauvegarder des visualisations dans `examples/`

### 3. Prédire sur une image personnalisée

```python
from predict import load_trained_model, predict_from_file

# Charger le modèle
model = load_trained_model()

# Prédire sur votre image
predicted, confidence = predict_from_file(model, 'chemin/vers/image.png')
```

## 🏗️ Architecture du modèle

Le CNN est composé de :
- **3 couches de convolution** avec MaxPooling (extraction de features)
- **1 couche dense** de 64 neurones
- **Dropout** (50%) pour éviter l'overfitting
- **Couche de sortie** avec softmax (10 classes)

```
Total params: ~100,000 paramètres
```

## 📊 Résultats obtenus

Le modèle a été entraîné et testé avec succès sur le dataset MNIST complet.

### Performances réelles :
```
✓ Exactitude sur test: 98.67%
✓ Temps d'entraînement: 27 secondes (CPU, 5 epochs)
✓ Dataset: 60,000 images train / 10,000 images test
✓ Prédictions correctes: 9,867 / 10,000
✓ Nombre d'erreurs: 133 seulement

Évolution de l'accuracy:
  Epoch 1: 97.09%
  Epoch 2: 97.83%
  Epoch 3: 98.42%
  Epoch 4: 98.67% ⭐ (meilleur)
  Epoch 5: 98.63%
```

### 🎯 Test sur échantillons aléatoires :
- 19/20 prédictions correctes (95%)
- Confiance moyenne: 99% sur prédictions correctes
- Erreurs principalement sur chiffres visuellement similaires (9↔8, 6↔0)

## 🔧 Fonctionnalités

- ✅ Téléchargement automatique du dataset MNIST
- ✅ Prétraitement et normalisation des images
- ✅ Architecture CNN optimisée
- ✅ Callbacks (EarlyStopping, ModelCheckpoint)
- ✅ Visualisation de l'entraînement (courbes accuracy/loss)
- ✅ Prédictions avec confiance et probabilités
- ✅ Support d'images personnalisées
- ✅ Évaluation complète sur dataset de test

## 📝 Exemples de code

### Charger les données
```python
from data_loader import prepare_mnist_dataset

X_train, y_train, X_test, y_test = prepare_mnist_dataset()
```

### Créer et entraîner le modèle
```python
from model import create_cnn_model, compile_model
from train import train_model

model = create_cnn_model()
model = compile_model(model)
history = train_model(model, X_train, y_train, X_test, y_test, epochs=10)
```

### Faire une prédiction
```python
from predict import load_trained_model, predict_single_image

model = load_trained_model()
predicted, confidence, probs = predict_single_image(model, image)
print(f"Chiffre prédit: {predicted} (confiance: {confidence:.1f}%)")
```

## 🎓 Concepts utilisés

- **Deep Learning** : Réseaux de neurones profonds
- **CNN** : Convolution pour détecter des patterns visuels
- **Data Augmentation** : Amélioration de la robustesse
- **Callbacks** : Optimisation de l'entraînement
- **Régularisation** : Dropout pour éviter l'overfitting

## 📈 Améliorations possibles

- [ ] Data augmentation (rotation, zoom, décalage)
- [ ] Tester différentes architectures (ResNet, VGG)
- [ ] Interface graphique pour dessiner et prédire
- [ ] Support de lettres (pas seulement chiffres)
- [ ] Déploiement web avec Flask/FastAPI
- [ ] Application mobile

## 👤 Auteur

**Ilyes Elhamdi**
- LinkedIn: [ilyes-elhamdi](https://www.linkedin.com/in/ilyes-elhamdi-320202248)
- Email: ilyeshamdi48@gmail.com

## 📄 Licence

Projet personnel - libre d'utilisation à des fins éducatives

## 🙏 Remerciements

- Dataset MNIST : Yann LeCun et al.
- TensorFlow/Keras pour le framework
