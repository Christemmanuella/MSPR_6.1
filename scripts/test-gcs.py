from google.cloud import storage


credentials_path = "E:/MSPR_6.1_WildLens/secrets/gcs-key.json"
bucket_name = "wildlens-footprint"

storage_client = storage.Client.from_service_account_json(credentials_path)
bucket = storage_client.get_bucket(bucket_name)
print(f"Connexion réussie au bucket : {bucket_name}")


blobs = list(bucket.list_blobs())
if blobs:
    for blob in blobs:
        print(f"Fichier trouvé : {blob.name}")
else:
    print("Aucun fichier trouvé dans le bucket.")