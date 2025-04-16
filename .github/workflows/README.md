# WildLens ETL Pipeline (MSPR_6.1)

## Description
Ce projet constitue la première étape (MSPR_6.1) du projet **WildLens**, une application visant à identifier les empreintes animales pour sensibiliser à la préservation de la faune sauvage. Ce pipeline ETL (Extract, Transform, Load) collecte, traite et stocke des données d’empreintes animales pour une utilisation future dans une application web et un modèle d’IA.

Le pipeline extrait des images d’empreintes (252 images, 13 espèces) depuis Google Cloud Storage (GCS), traite les images (redimensionnement, suppression de filigranes, suppression des doublons), et stocke les métadonnées dans une base de données PostgreSQL sur Supabase.

## Structure du projet
- `scripts/` : Contient les scripts Python pour le pipeline ETL.
  - `etl_pipeline.py` : Script principal qui orchestre l’extraction, le traitement et le chargement des données.
- `.github/workflows/main.yml` : Pipeline CI pour vérifier l’exécution du script ETL.
- `requirements.txt` : Liste des dépendances Python nécessaires.
- `.gitignore` : Exclut les fichiers sensibles et inutiles (`.env`, données brutes, etc.).

## Prérequis
- Python 3.8+
- Dépendances : `google-cloud-storage`, `supabase`, `pandas`, `Pillow`
- Compte Google Cloud avec accès au bucket `wildlens-footprint`
- Compte Supabase avec une base de données PostgreSQL configurée

## Installation
1. Clonez le dépôt :
