"""
Script pour charger et préparer le dataset MNIST
MNIST = dataset de 70,000 images de chiffres manuscrits (0-9)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_mnist_data():
    """
    Charge le dataset MNIST depuis Keras
    Contient 60,000 images d'entraînement et 10,000 images de test
    """
    print("Chargement du dataset MNIST...")
    
    # Télécharger et charger le dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"✓ Dataset chargé:")
    print(f"  - Entraînement: {len(X_train)} images")
    print(f"  - Test: {len(X_test)} images")
    print(f"  - Taille des images: {X_train.shape[1]}x{X_train.shape[2]} pixels")
    
    return (X_train, y_train), (X_test, y_test)


def preprocess_data(X_train, y_train, X_test, y_test):
    """
    Prétraite les données pour l'entraînement du réseau de neurones
    - Normalise les valeurs de pixels entre 0 et 1
    - Redimensionne pour le CNN
    - Convertit les labels en format one-hot
    """
    print("\nPrétraitement des données...")
    
    # Normaliser les pixels (de 0-255 à 0-1)
    # Cela aide le réseau à apprendre plus vite
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Ajouter une dimension pour le canal (grayscale)
    # Shape: (28, 28) -> (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Convertir les labels en format one-hot
    # Exemple: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"✓ Données prétraitées:")
    print(f"  - Shape X_train: {X_train.shape}")
    print(f"  - Shape y_train: {y_train.shape}")
    print(f"  - Valeurs pixels: {X_train.min():.1f} à {X_train.max():.1f}")
    
    return X_train, y_train, X_test, y_test


def show_sample_images(X_train, y_train, n_images=10):
    """
    Affiche quelques images d'exemple du dataset
    Utile pour visualiser les données
    """
    print(f"\nAffichage de {n_images} images d'exemple...")
    
    # Créer une grille d'images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i in range(n_images):
        # Prendre une image aléatoire
        idx = np.random.randint(0, len(X_train))
        image = X_train[idx].squeeze()  # Enlever la dimension du canal
        label = np.argmax(y_train[idx])  # Convertir one-hot en nombre
        
        # Afficher l'image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('../examples/sample_images.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("✓ Images sauvegardées dans 'examples/sample_images.png'")


def get_data_statistics(y_train, y_test):
    """
    Affiche des statistiques sur la distribution des chiffres
    """
    print("\n=== Statistiques du dataset ===")
    
    # Convertir one-hot en labels numériques
    train_labels = np.argmax(y_train, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    
    print("\nDistribution des chiffres (entraînement):")
    for digit in range(10):
        count = np.sum(train_labels == digit)
        print(f"  Chiffre {digit}: {count} images")
    
    print("\nDistribution des chiffres (test):")
    for digit in range(10):
        count = np.sum(test_labels == digit)
        print(f"  Chiffre {digit}: {count} images")


# Fonction principale pour charger et préparer tout
def prepare_mnist_dataset(show_samples=True):
    """
    Pipeline complet de préparation du dataset
    """
    print("=" * 60)
    print("PRÉPARATION DU DATASET MNIST")
    print("=" * 60)
    
    # Charger les données
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    
    # Prétraiter
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Afficher des exemples
    if show_samples:
        show_sample_images(X_train, y_train)
    
    # Statistiques
    get_data_statistics(y_train, y_test)
    
    print("\n" + "=" * 60)
    print("✓ Dataset prêt pour l'entraînement")
    print("=" * 60)
    
    return X_train, y_train, X_test, y_test


# Test du script si exécuté directement
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_mnist_dataset(show_samples=True)
