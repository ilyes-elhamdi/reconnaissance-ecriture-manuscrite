"""
Création du modèle CNN pour la reconnaissance de chiffres manuscrits
CNN = Convolutional Neural Network (Réseau de Neurones Convolutif)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Crée un modèle CNN simple mais efficace pour MNIST
    
    Architecture:
    - 2 blocs de convolution + pooling
    - 1 couche dense
    - 1 couche de sortie avec softmax
    """
    print("\nCréation du modèle CNN...")
    
    model = models.Sequential([
        # Premier bloc de convolution
        # Détecte les features basiques (lignes, courbes)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Deuxième bloc de convolution
        # Détecte des patterns plus complexes
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Troisième bloc de convolution
        # Capture les détails fins des chiffres
        layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        
        # Aplatir les features pour le réseau dense
        layers.Flatten(name='flatten'),
        
        # Couche dense pour apprendre les combinaisons
        layers.Dense(64, activation='relu', name='dense1'),
        
        # Dropout pour éviter l'overfitting
        layers.Dropout(0.5, name='dropout'),
        
        # Couche de sortie (10 classes pour les chiffres 0-9)
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    print("✓ Modèle créé avec succès")
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile le modèle avec l'optimiseur et la fonction de perte
    """
    print("\nCompilation du modèle...")
    
    model.compile(
        # Optimiseur Adam - ajuste automatiquement le learning rate
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        
        # Loss function pour classification multi-classes
        loss='categorical_crossentropy',
        
        # Métriques à suivre pendant l'entraînement
        metrics=['accuracy']
    )
    
    print("✓ Modèle compilé")
    return model


def get_model_summary(model):
    """
    Affiche un résumé détaillé du modèle
    """
    print("\n" + "=" * 60)
    print("ARCHITECTURE DU MODÈLE")
    print("=" * 60)
    model.summary()
    
    # Calculer le nombre total de paramètres
    total_params = model.count_params()
    print(f"\n✓ Nombre total de paramètres: {total_params:,}")
    print("=" * 60)


def setup_callbacks(model_save_path='../models/best_model.keras'):
    """
    Configure les callbacks pour l'entraînement
    - EarlyStopping: arrête si plus de progrès
    - ModelCheckpoint: sauvegarde le meilleur modèle
    """
    print("\nConfiguration des callbacks...")
    
    callbacks = [
        # Arrêter l'entraînement si pas d'amélioration après 3 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Sauvegarder le meilleur modèle automatiquement
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configurés")
    return callbacks


def create_simple_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Version simplifiée du modèle pour tests rapides
    Moins de paramètres = entraînement plus rapide
    """
    print("\nCréation du modèle simple (version rapide)...")
    
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("✓ Modèle simple créé")
    return model


# Test du script
if __name__ == "__main__":
    print("=" * 60)
    print("TEST DE CRÉATION DU MODÈLE")
    print("=" * 60)
    
    # Créer le modèle complet
    model = create_cnn_model()
    model = compile_model(model)
    get_model_summary(model)
    
    # Créer le modèle simple
    print("\n" + "=" * 60)
    simple_model = create_simple_model()
    simple_model = compile_model(simple_model)
    get_model_summary(simple_model)
    
    print("\n✓ Les deux modèles sont prêts à être entraînés")
