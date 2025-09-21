# 👗🧥 Pipeline IA Tenues/Vêtements – CLIP, ViT, SegFormer, Weaviate & Streamlit 🛠️

## Présentation

Ce projet propose un pipeline complet pour :
- **Uploader un ZIP d’images de tenues (images qui serviront de référenciel "tendance mode"**
- Détecter, segmenter et cropper automatiquement chaque vêtement (SegFormer)
- Générer leurs embeddings (FashionCLIP + ViT)
- Organiser les résultats (fichiers, labels, embeddings, CSV…)
- Sauvegarder les prédictions et *populer une base Weaviate* (multivector/fusion recherche sémantique/image)
- Visualiser tout le pipeline dans une application **Streamlit** simple

Convient aux besoins R&D (créa dataset, test modèles, moteur de recherche, etc).

---

## Fonctionnalités

- **Upload ZIP** de photos de tenues (mode dataset fashion)
- **Détection automatique** des vêtements (SegFormer spécialisé habillement)
- **Découpe et labellisation textile directe** (haut/bas/robe/pantalon, etc.)
- **Embeddings multimodaux** (FashionCLIP + ViT) prêts pour vector-db
- **Export fichiers, CSV, crops**
- **Insertion base Weaviate** (collections [Tenues](#collections) & [Vetements](#collections))
- **Dashboard visuel Interactif (Streamlit)**

---

## Installation

### Prérequis

- Python 3.9+
- (Optionnel) Accès GPU/CUDA si vous avez un GPU (fortement conseillé !)
- pip installé

### Installation des dépendances

