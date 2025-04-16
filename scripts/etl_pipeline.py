import os
import tempfile
import shutil
from datetime import datetime, timezone
from google.cloud import storage
from supabase import create_client, Client
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration des constantes
SERVICE_ACCOUNT_KEY_PATH = "../secrets/gcs-key.json"
BUCKET_NAME = "wildlens-footprint"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = (224, 224)
MAMMALS_PATH = "Mammifères/"

# Initialisation du client Supabase
def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Variables SUPABASE_URL ou SUPABASE_SERVICE_ROLE_KEY manquantes dans .env.")
    return create_client(supabase_url, supabase_key)

# Initialisation du client Google Cloud Storage
def get_storage_client():
    return storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY_PATH)

# Vérifier si les tables sont vides
def check_tables_empty(supabase: Client):
    try:
        infos_count = supabase.table("infos_especes").select("*").execute().data
        images_count = supabase.table("footprint_images").select("*").execute().data
        return len(infos_count) == 0 and len(images_count) == 0
    except Exception as e:
        print(f"Erreur lors de la vérification des tables vides : {e}")
        raise

# Réinitialiser les tables
def reset_tables(supabase: Client):
    try:
        supabase.table("footprint_images").delete().neq("id", -1).execute()
        supabase.table("infos_especes").delete().neq("id", -1).execute()
        supabase.table("data_quality_log").delete().neq("id", -1).execute()
        print("Les tables infos_especes et footprint_images sont vides. Réinitialisation des DEUX tables...")
    except Exception as e:
        print(f"Erreur lors de la réinitialisation des tables : {e}")
        raise

# Vider la table footprint_images
def clear_footprint_images(supabase: Client):
    """Vide la table footprint_images avant de relancer le traitement."""
    try:
        supabase.table("footprint_images").delete().neq("id", -1).execute()
        print("Table footprint_images vidée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la suppression des données dans footprint_images : {e}")

# Remplir la table infos_especes à partir de infos_especes.xlsx
def ensure_infos_especes_filled(supabase: Client, bucket):
    # Vérifier si la table est vide
    response = supabase.table("infos_especes").select("*").execute()
    if response.data:
        print("La table infos_especes contient déjà des données. Saut du remplissage.")
        return

    # Télécharger le fichier principal infos_especes.xlsx
    main_blob = bucket.blob("infos_especes.xlsx")
    if not main_blob.exists():
        print("Erreur : Le fichier infos_especes.xlsx n'existe pas dans le bucket wildlens-footprint.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="excel_files_")
    main_local_path = os.path.join(tmp_dir, "infos_especes.xlsx")
    try:
        main_blob.download_to_filename(main_local_path)
        df_main = pd.read_excel(main_local_path)
    except Exception as e:
        print(f"Erreur lors du téléchargement ou de la lecture de infos_especes.xlsx : {e}")
        return
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Insérer les données dans la table infos_especes
    required_columns = [
        "Espèce", "Nom latin", "Famille", "Région", "Habitat", "Fun fact", "Description"
    ]
    if not all(col in df_main.columns for col in required_columns):
        print(f"Erreur : Les colonnes requises {required_columns} ne sont pas toutes présentes dans infos_especes.xlsx.")
        return

    data_to_insert = df_main[required_columns].to_dict("records")
    try:
        supabase.table("infos_especes").insert(data_to_insert).execute()
        print("Insertion de toutes les lignes dans infos_especes (table vide initialement).")
    except Exception as e:
        print(f"Erreur lors de l'insertion dans infos_especes : {e}")
        return

# Traiter et redimensionner une image
def process_image(image_data: bytes) -> bytes:
    try:
        image = Image.open(BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Recadrer le haut de l'image pour supprimer le filigrane (10% supérieur)
        width, height = image.size
        crop_height = int(height * 0.1)  # Recadrer les 10% supérieurs
        image = image.crop((0, crop_height, width, height))  # (left, upper, right, lower)
        
        # Redimensionner l'image après recadrage
        image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG", quality=95)
        return output_buffer.getvalue()
    except Exception as e:
        print(f"Erreur lors du traitement de l'image : {e}")
        return None

# Préparer les métadonnées de l'image et traiter l'image sans uploader immédiatement
def prepare_image_data(blob, bucket, species_id, processed_folder="processed_data"):
    try:
        image_data = blob.download_as_bytes()
        processed_data = process_image(image_data)
        if processed_data is None:
            print(f"Image corrompue ou non traitable : {blob.name}")
            return None

        # Construire le chemin de destination
        relative_path = blob.name[len(MAMMALS_PATH):]
        species_name = relative_path.split("/")[0]
        processed_path = f"{processed_folder}/{species_name}/{os.path.basename(blob.name)}"

        # Générer l'URL publique (sera valide après upload)
        public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{processed_path}"
        return {
            "espece_id": species_id,
            "image_url": public_url,
            "path": blob.name,
            "processed_path": processed_path,
            "image_name": os.path.basename(blob.name),
            "processed_data": processed_data  # Données de l'image traitée
        }
    except Exception as e:
        print(f"Erreur lors du traitement de {blob.name} : {e}")
        return None

# Uploader une image dans GCS
def upload_image(bucket, processed_path, processed_data):
    try:
        processed_blob = bucket.blob(processed_path)
        processed_blob.upload_from_string(processed_data, content_type="image/jpeg")
    except Exception as e:
        print(f"Erreur lors de l'upload de {processed_path} : {e}")
        return False
    return True

# Récupérer l'ID d'une espèce à partir de son nom
def get_species_id(supabase: Client, species_name: str) -> int:
    response = supabase.table("infos_especes").select("id").eq("Espèce", species_name).execute()
    if response.data:
        return response.data[0]["id"]
    return None

# Traiter toutes les images dans Mammifères/
def process_images(supabase: Client, bucket):
    blobs = bucket.list_blobs(prefix=MAMMALS_PATH)
    images_to_insert = []
    seen_image_keys = set()  # Pour suivre les paires species_name/image_name

    for blob in blobs:
        if not any(blob.name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            continue

        # Extraire le nom de l'espèce à partir du chemin (Mammifères/Espèce/...)
        species_name = blob.name.split("/")[1]
        species_id = get_species_id(supabase, species_name)

        if species_id is None:
            print(f"Espèce '{species_name}' non trouvée dans la base. Saut de {blob.name}.")
            continue

        # Vérifier les doublons en utilisant species_name/image_name
        image_name = os.path.basename(blob.name)
        image_key = f"{species_name}/{image_name}"
        if image_key in seen_image_keys:
            print(f"Doublon détecté : {image_key}. Saut de l'image.")
            continue

        # Traiter l'image et préparer les métadonnées
        image_data = prepare_image_data(blob, bucket, species_id)
        if image_data:
            # Uploader l'image uniquement si elle n'est pas un doublon
            if upload_image(bucket, image_data["processed_path"], image_data["processed_data"]):
                seen_image_keys.add(image_key)
                # Retirer processed_data des métadonnées avant insertion dans la base
                image_data_to_insert = {k: v for k, v in image_data.items() if k != "processed_data"}
                images_to_insert.append(image_data_to_insert)

    # Insérer les données dans la table footprint_images
    if images_to_insert:
        try:
            supabase.table("footprint_images").insert(images_to_insert).execute()
        except Exception as e:
            print(f"Erreur lors de l'insertion dans footprint_images : {e}")
            return

    print("Traitement des images terminé.")

# Tests de qualité des données (QDD)
def run_data_quality_checks_and_log(supabase: Client, bucket):
    # Vérifier infos_especes
    infos_data = supabase.table("infos_especes").select("*").execute().data
    infos_count = len(infos_data)
    infos_completeness = 1 if infos_count >= 13 else 0
    infos_consistency = 1
    infos_accuracy = 1

    # Vérifier les colonnes non nulles
    required_columns = ["Espèce", "Nom latin", "Famille", "Région", "Habitat", "Fun fact", "Description"]
    for row in infos_data:
        for col in required_columns:
            if pd.isna(row[col]) or row[col] == "":
                infos_consistency = 0
                break

    infos_errors = []
    if infos_completeness == 0:
        infos_errors.append(f"Exhaustivité : Seulement {infos_count} lignes dans infos_especes, attendu >= 13.")

    # Vérifier footprint_images
    images_data = supabase.table("footprint_images").select("*").execute().data
    images_count = len(images_data)
    blobs = list(bucket.list_blobs(prefix=MAMMALS_PATH))
    gcs_images = [blob.name for blob in blobs if any(blob.name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)]
    gcs_images_count = len(gcs_images)
    
    # Initialiser images_errors
    images_errors = []
    images_completeness = 1
    if images_count != gcs_images_count:
        images_completeness = 0
        db_image_names = {row["image_name"] for row in images_data}
        gcs_image_names = {os.path.basename(img) for img in gcs_images}
        
        # Images présentes dans GCS mais absentes dans la base
        missing_in_db = list(gcs_image_names - db_image_names)
        # Images présentes dans la base mais absentes dans GCS
        extra_in_db = list(db_image_names - gcs_image_names)
        
        if missing_in_db:
            images_errors.append(f"Images manquantes dans la base : {missing_in_db}")
        if extra_in_db:
            images_errors.append(f"Images supplémentaires dans la base : {extra_in_db}")
    
    images_consistency = 1
    images_accuracy = 1

    # Vérifier que les espece_id existent dans infos_especes
    species_ids = {row["id"] for row in supabase.table("infos_especes").select("id").execute().data}
    for img in images_data:
        if img["espece_id"] not in species_ids:
            images_consistency = 0
            images_errors.append(f"Incohérence : espece_id {img['espece_id']} non trouvé dans infos_especes pour l'image {img['image_name']}")
            break

    # Enregistrer dans data_quality_log
    run_timestamp = datetime.now(timezone.utc).isoformat()
    log_entry_infos = {
        "table_name": "infos_especes",
        "completeness": infos_completeness,
        "consistency": infos_consistency,
        "accuracy": infos_accuracy,
        "errors": ", ".join(infos_errors) if infos_errors else "Aucun problème détecté",
        "run_timestamp": run_timestamp
    }
    log_entry_images = {
        "table_name": "footprint_images",
        "completeness": images_completeness,
        "consistency": images_consistency,
        "accuracy": images_accuracy,
        "errors": ", ".join(images_errors) if images_errors else "Aucun problème détecté",
        "run_timestamp": run_timestamp
    }

    supabase.table("data_quality_log").insert([log_entry_infos, log_entry_images]).execute()

    print(f"Qualité des données pour infos_especes : [{infos_completeness}, {infos_consistency}, {infos_accuracy}], Erreurs : {log_entry_infos['errors']}")
    print(f"Qualité des données pour footprint_images : [{images_completeness}, {images_consistency}, {images_accuracy}], Erreurs : {log_entry_images['errors']}")

# Fonction principale
def main():
    # Initialisation des clients
    supabase = get_supabase_client()
    storage_client = get_storage_client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Vider footprint_images avant de commencer
    clear_footprint_images(supabase)

    # Vérifier et réinitialiser les tables si elles sont vides
    try:
        if check_tables_empty(supabase):
            reset_tables(supabase)
    except Exception as e:
        # Enregistrer l'erreur dans data_quality_log
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "ETL_process",
            "test_results": [0],
            "error_description": f"Erreur dans maybe_reset_tables : {str(e)}",
            "execution_time": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        print(f"Erreur dans le processus ETL : {e}")
        return

    # Remplir infos_especes
    ensure_infos_especes_filled(supabase, bucket)

    # Traiter les images
    process_images(supabase, bucket)

    # Exécuter les tests de qualité des données
    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_data_quality_checks_and_log(supabase, bucket)

if __name__ == "__main__":
    main()