import os
import tempfile
import shutil
from datetime import datetime, timezone
from google.cloud import storage
from supabase import create_client, Client
from PIL import Image
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Définition des constantes pour la configuration du pipeline ETL
SERVICE_ACCOUNT_KEY_PATH = "../secrets/gcs-key.json"
BUCKET_NAME = "wildlens-footprint"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = (225, 225)
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

# Vérification de l’état des tables `infos_especes` et `footprint_images`
def check_tables_empty(supabase: Client):
    try:
        infos_count = supabase.table("infos_especes").select("*").execute().data
        images_count = supabase.table("footprint_images").select("*").execute().data
        return len(infos_count) == 0 and len(images_count) == 0
    except Exception as e:
        raise Exception(f"Erreur lors de la vérification des tables vides : {e}")

# Réinitialisation des tables si elles sont vides (sans vider data_quality_log)
def reset_tables(supabase: Client):
    try:
        supabase.table("footprint_images").delete().neq("id", -1).execute()
        supabase.table("infos_especes").delete().neq("id", -1).execute()
    except Exception as e:
        raise Exception(f"Erreur lors de la réinitialisation des tables : {e}")

# Vidage de la table `footprint_images` avant chaque exécution
def clear_footprint_images(supabase: Client):
    try:
        supabase.table("footprint_images").delete().neq("id", -1).execute()
    except Exception as e:
        raise Exception(f"Erreur lors de la suppression des données dans footprint_images : {e}")

# Remplissage de la table `infos_especes` avec les données de `infos_especes.xlsx`
def ensure_infos_especes_filled(supabase: Client, bucket, iteration: int):
    response = supabase.table("infos_especes").select("*").execute()
    if response.data:
        return len(response.data)

    main_blob = bucket.blob("infos_especes.xlsx")
    if not main_blob.exists():
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "infos_especes",
            "completeness": 0,
            "consistency": 1,
            "accuracy": 1,
            "errors": "Exhaustivité: Fichier infos_especes.xlsx introuvable dans le bucket GCS",
            "run_timestamp": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        return 0

    tmp_dir = tempfile.mkdtemp(prefix="excel_files_")
    main_local_path = os.path.join(tmp_dir, "infos_especes.xlsx")
    try:
        main_blob.download_to_filename(main_local_path)
        df_main = pd.read_excel(main_local_path)
    except Exception as e:
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "infos_especes",
            "completeness": 0,
            "consistency": 1,
            "accuracy": 1,
            "errors": f"Exhaustivité: Erreur lors de la lecture de infos_especes.xlsx",
            "run_timestamp": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    required_columns = [
        "Espèce", "Nom latin", "Famille", "Région", "Habitat", "Fun fact", "Description"
    ]
    if not all(col in df_main.columns for col in required_columns):
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "infos_especes",
            "completeness": 0,
            "consistency": 1,
            "accuracy": 1,
            "errors": f"Exhaustivité: Colonnes manquantes dans infos_especes.xlsx",
            "run_timestamp": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        return 0

    data_to_insert = df_main[required_columns].to_dict("records")

    # Simuler une erreur de complétude : insérer moins de lignes dans certaines itérations
    if iteration % 3 == 0:
        data_to_insert = data_to_insert[:max(1, len(data_to_insert)//2)]

    try:
        supabase.table("infos_especes").insert(data_to_insert).execute()
        return len(data_to_insert)
    except Exception as e:
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "infos_especes",
            "completeness": 0,
            "consistency": 1,
            "accuracy": 1,
            "errors": f"Exhaustivité: Erreur lors de l’insertion dans infos_especes",
            "run_timestamp": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        return 0

# Traitement et redimensionnement d’une image
def process_image(image_data: bytes) -> bytes:
    try:
        image = Image.open(BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        width, height = image.size
        crop_height = int(height * 0.1)
        image = image.crop((0, crop_height, width, height))
        
        image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG", quality=95)
        return output_buffer.getvalue()
    except Exception as e:
        raise Exception(f"Erreur lors du traitement de l’image : {e}")

# Préparation des métadonnées, traitement de l’image et conversion en tableau NumPy
def prepare_image_data(blob, bucket, species_id, supabase: Client, iteration: int, processed_folder="processed_data"):
    try:
        image_data = blob.download_as_bytes()
        processed_data = process_image(image_data)
        if processed_data is None:
            run_timestamp = datetime.now(timezone.utc).isoformat()
            log_entry = {
                "table_name": "footprint_images",
                "completeness": 0,
                "consistency": 1,
                "accuracy": 1,
                "errors": f"Image corrompue ou non traitable",
                "run_timestamp": run_timestamp
            }
            supabase.table("data_quality_log").insert(log_entry).execute()
            return None

        image = Image.open(BytesIO(processed_data)).convert("L")
        numpy_array = np.array(image)

        relative_path = blob.name[len(MAMMALS_PATH):]
        species_name = relative_path.split("/")[0]
        numpy_text_path = f"{processed_folder}/numpy_text/{species_name}/{os.path.basename(blob.name)}_numpy.txt"
        with open("temp_numpy.txt", "w") as f:
            f.write("Extrait 5x5 du tableau NumPy (valeurs des pixels en niveaux de gris) :\n")
            f.write(np.array2string(numpy_array[:5, :5], separator=", "))
            f.write(f"\n\nForme du tableau : {numpy_array.shape}")
        bucket.blob(numpy_text_path).upload_from_filename("temp_numpy.txt")

        numpy_text_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{numpy_text_path}"

        processed_path = f"{processed_folder}/{species_name}/{os.path.basename(blob.name)}"
        public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{processed_path}"

        # Simuler une URL inaccessible dans certaines itérations
        if iteration % 5 == 0:
            public_url = f"https://invalid-url-{iteration}.com"

        return {
            "espece_id": species_id,
            "image_url": public_url,
            "path": blob.name,
            "processed_path": processed_path,
            "image_name": os.path.basename(blob.name),
            "processed_data": processed_data,
            "numpy_text_url": numpy_text_url
        }
    except Exception as e:
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "footprint_images",
            "completeness": 0,
            "consistency": 1,
            "accuracy": 1,
            "errors": f"Erreur lors du traitement de l’image",
            "run_timestamp": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        return None

# Upload de l’image transformée dans GCS
def upload_image(bucket, processed_path, processed_data, supabase: Client):
    try:
        processed_blob = bucket.blob(processed_path)
        processed_blob.upload_from_string(processed_data, content_type="image/jpeg")
        return True
    except Exception as e:
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "footprint_images",
            "completeness": 0,
            "consistency": 1,
            "accuracy": 1,
            "errors": f"Erreur lors de l’upload de l’image",
            "run_timestamp": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        return False

# Récupération de l’ID d’une espèce à partir de son nom
def get_species_id(supabase: Client, species_name: str) -> int:
    response = supabase.table("infos_especes").select("id").eq("Espèce", species_name).execute()
    if response.data:
        return response.data[0]["id"]
    return None

# Traitement des images dans `Mammifères/`
def process_images(supabase: Client, bucket, iteration: int):
    blobs = list(bucket.list_blobs(prefix=MAMMALS_PATH))
    valid_blobs = [blob for blob in blobs if any(blob.name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)]
    if not valid_blobs:
        run_timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "table_name": "footprint_images",
            "completeness": 0,
            "consistency": 1,
            "accuracy": 1,
            "errors": "Aucune image trouvée dans Mammifères/ au format .jpg, .jpeg ou .png",
            "run_timestamp": run_timestamp
        }
        supabase.table("data_quality_log").insert(log_entry).execute()
        return set()

    images_to_insert = []
    seen_image_keys = set()
    missing_species = set()

    for blob in valid_blobs:
        species_name = blob.name.split("/")[1]
        species_id = get_species_id(supabase, species_name)
        if species_id is None:
            missing_species.add(species_name)
            continue

        image_name = os.path.basename(blob.name)
        image_key = f"{species_name}/{image_name}"
        if image_key in seen_image_keys:
            continue

        image_data = prepare_image_data(blob, bucket, species_id, supabase, iteration)
        if image_data and upload_image(bucket, image_data["processed_path"], image_data["processed_data"], supabase):
            seen_image_keys.add(image_key)
            image_data_to_insert = {k: v for k, v in image_data.items() if k != "processed_data"}
            images_to_insert.append(image_data_to_insert)

    # Simuler une erreur de complétude : insérer moins d’images dans certaines itérations
    if iteration % 3 == 1 and images_to_insert:
        images_to_insert = images_to_insert[:max(1, len(images_to_insert)//2)]

    if images_to_insert:
        try:
            supabase.table("footprint_images").insert(images_to_insert).execute()
        except Exception as e:
            raise Exception(f"Erreur lors de l’insertion dans footprint_images : {e}")

    return missing_species

# Exécution des tests de qualité des données (QDD)
def run_data_quality_checks_and_log(supabase: Client, bucket, missing_species, expected_infos_count, iteration: int):
    # Vérifier infos_especes
    infos_data = supabase.table("infos_especes").select("*").execute().data
    infos_count = len(infos_data)
    infos_completeness = 1 if infos_count >= 13 else 0
    infos_consistency = 1
    infos_accuracy = 1
    infos_errors = []

    # Vérifier la complétude (nombre de lignes attendu)
    if infos_completeness == 0:
        infos_errors.append(f"Exhaustivité: Seulement {infos_count} lignes dans infos_especes, attendu >= 13")

    # Vérifier la cohérence (valeurs nulles ou vides)
    required_columns = ["Espèce", "Nom latin", "Famille", "Région", "Habitat", "Fun fact", "Description"]
    for row in infos_data:
        for col in required_columns:
            if pd.isna(row[col]) or row[col] == "":
                infos_consistency = 0
                infos_errors.append("Cohérence: Valeurs manquantes ou vides")
                break
        if infos_consistency == 0:
            break

    # Vérifier footprint_images
    images_data = supabase.table("footprint_images").select("*").execute().data
    images_count = len(images_data)
    blobs = list(bucket.list_blobs(prefix=MAMMALS_PATH))
    gcs_images = [blob.name for blob in blobs if any(blob.name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)]
    gcs_images_count = len(gcs_images)

    images_completeness = 1
    images_consistency = 1
    images_accuracy = 1
    images_errors = []

    # Vérifier la complétude
    if images_count != gcs_images_count:
        images_completeness = 0
        images_errors.append(f"Exhaustivité: Seulement {images_count} images sur {gcs_images_count} attendues")

    # Vérifier la cohérence (espece_id existe dans infos_especes)
    species_ids = {row["id"] for row in supabase.table("infos_especes").select("id").execute().data}
    for img in images_data:
        if img["espece_id"] not in species_ids:
            images_consistency = 0
            images_errors.append("Incohérence: espece_id non trouvé dans infos_especes")
            break

    # Vérifier l’exactitude (URLs accessibles)
    if images_data:
        sample_urls = [img["image_url"] for img in images_data[:5]]
        for url in sample_urls:
            try:
                response = requests.head(url, timeout=5)
                if response.status_code != 200:
                    images_accuracy = 0
                    images_errors.append(f"Exactitude: URL inaccessible")
                    break
            except Exception:
                images_accuracy = 0
                images_errors.append(f"Exactitude: Erreur lors de la vérification de l’URL")
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

# Fonction principale du pipeline ETL
def main():
    try:
        supabase = get_supabase_client()
        storage_client = get_storage_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Boucle pour générer 24 enregistrements (12 exécutions * 2 = 24)
        for i in range(12):
            clear_footprint_images(supabase)
            expected_infos_count = ensure_infos_especes_filled(supabase, bucket, i)
            missing_species = process_images(supabase, bucket, i)
            run_data_quality_checks_and_log(supabase, bucket, missing_species, expected_infos_count, i)

        print("Traitement d'images terminé")
    except Exception as e:
        print(f"Erreur dans le processus ETL : {str(e)}")

if __name__ == "__main__":
    main()