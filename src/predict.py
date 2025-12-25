"""
Script de prédiction pour tester le modèle sur des images
Permet de visualiser les prédictions du modèle
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image


def load_trained_model(model_path='../models/mnist_cnn_full_final.keras'):
    """
    Charge le modèle entraîné depuis un fichier
    """
    print(f"Chargement du modèle depuis '{model_path}'...")
    
    try:
        model = load_model(model_path)
        print("✓ Modèle chargé avec succès")
        return model
    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        print("Astuce: Entraînez d'abord le modèle avec 'python train.py'")
        return None


def predict_single_image(model, image):
    """
    Fait une prédiction sur une seule image
    Retourne le chiffre prédit et les probabilités pour chaque classe
    """
    # S'assurer que l'image a la bonne forme
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)  # Ajouter dimension canal
    
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)  # Ajouter dimension batch
    
    # Normaliser si nécessaire
    if image.max() > 1.0:
        image = image.astype('float32') / 255.0
    
    # Faire la prédiction
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    return predicted_class, confidence, predictions[0]


def visualize_prediction(image, true_label, predicted_label, confidence, probabilities):
    """
    Visualise l'image avec la prédiction et les probabilités
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Afficher l'image
    ax1.imshow(image.squeeze(), cmap='gray')
    
    # Titre avec couleur selon si c'est correct ou non
    is_correct = (true_label == predicted_label)
    color = 'green' if is_correct else 'red'
    title = f"Vrai: {true_label} | Prédit: {predicted_label}\nConfiance: {confidence:.1f}%"
    ax1.set_title(title, color=color, fontweight='bold')
    ax1.axis('off')
    
    # Graphique des probabilités
    ax2.bar(range(10), probabilities * 100, color='skyblue')
    ax2.bar(predicted_label, probabilities[predicted_label] * 100, color='green')
    ax2.set_xlabel('Chiffre')
    ax2.set_ylabel('Probabilité (%)')
    ax2.set_title('Probabilités pour chaque chiffre')
    ax2.set_xticks(range(10))
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def test_on_random_samples(model, n_samples=10):
    """
    Teste le modèle sur des échantillons aléatoires du dataset de test
    """
    print(f"\n=== Test sur {n_samples} échantillons aléatoires ===")
    
    # Charger le dataset de test
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.astype('float32') / 255.0
    
    # Sélectionner des images aléatoires
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    correct = 0
    
    for i, idx in enumerate(indices):
        image = X_test[idx]
        true_label = y_test[idx]
        
        # Prédire
        predicted_label, confidence, probabilities = predict_single_image(model, image)
        
        # Afficher le résultat
        is_correct = (predicted_label == true_label)
        status = "✓" if is_correct else "✗"
        print(f"{status} Image {i+1}: Vrai={true_label}, Prédit={predicted_label}, Confiance={confidence:.1f}%")
        
        if is_correct:
            correct += 1
        
        # Visualiser quelques exemples
        if i < 5:
            fig = visualize_prediction(image, true_label, predicted_label, confidence, probabilities)
            fig.savefig(f'../examples/prediction_{i+1}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
    
    # Statistiques
    accuracy = (correct / n_samples) * 100
    print(f"\n✓ Accuracy sur les échantillons: {accuracy:.1f}% ({correct}/{n_samples})")


def predict_from_file(model, image_path):
    """
    Charge une image depuis un fichier et fait une prédiction
    Utile pour tester avec vos propres images de chiffres
    """
    print(f"\nPrédiction sur l'image: {image_path}")
    
    try:
        # Charger l'image
        img = Image.open(image_path).convert('L')  # Convertir en grayscale
        img = img.resize((28, 28))  # Redimensionner à 28x28
        
        # Convertir en array numpy
        image = np.array(img)
        
        # Inverser si nécessaire (MNIST a fond noir)
        if image.mean() > 127:
            image = 255 - image
        
        # Prédire
        predicted_label, confidence, probabilities = predict_single_image(model, image)
        
        print(f"✓ Chiffre prédit: {predicted_label}")
        print(f"✓ Confiance: {confidence:.1f}%")
        
        # Visualiser
        fig = visualize_prediction(image, "?", predicted_label, confidence, probabilities)
        fig.savefig('../examples/custom_prediction.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("✓ Résultat sauvegardé dans 'examples/custom_prediction.png'")
        
        return predicted_label, confidence
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None, None


def evaluate_on_full_test_set(model):
    """
    Évalue le modèle sur l'ensemble complet du dataset de test
    """
    print("\n=== Évaluation sur le dataset de test complet ===")
    
    # Charger les données de test
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.astype('float32') / 255.0
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Faire des prédictions
    print("Prédiction en cours...")
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculer l'accuracy
    correct = np.sum(predicted_labels == y_test)
    total = len(y_test)
    accuracy = (correct / total) * 100
    
    print(f"\n✓ Images testées: {total}")
    print(f"✓ Prédictions correctes: {correct}")
    print(f"✓ Accuracy: {accuracy:.2f}%")
    
    # Afficher quelques erreurs
    print("\n=== Exemples d'erreurs ===")
    errors = np.where(predicted_labels != y_test)[0]
    
    if len(errors) > 0:
        print(f"Nombre d'erreurs: {len(errors)}")
        
        # Afficher les 5 premières erreurs
        for i, idx in enumerate(errors[:5]):
            true_label = y_test[idx]
            pred_label = predicted_labels[idx]
            confidence = predictions[idx][pred_label] * 100
            print(f"  Erreur {i+1}: Vrai={true_label}, Prédit={pred_label}, Confiance={confidence:.1f}%")
    else:
        print("Aucune erreur trouvée!")
    
    return accuracy


# Mode interactif
if __name__ == "__main__":
    print("=" * 70)
    print("PRÉDICTION AVEC LE MODÈLE DE RECONNAISSANCE D'ÉCRITURE")
    print("=" * 70)
    
    # Charger le modèle
    model = load_trained_model()
    
    if model is not None:
        # Test sur échantillons aléatoires
        test_on_random_samples(model, n_samples=20)
        
        # Évaluation complète
        evaluate_on_full_test_set(model)
        
        print("\n" + "=" * 70)
        print("✓ Tests terminés! Vérifiez le dossier 'examples/' pour les images")
        print("=" * 70)
