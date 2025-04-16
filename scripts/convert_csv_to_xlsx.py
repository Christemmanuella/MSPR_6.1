import pandas as pd

# Charger le fichier CSV
df = pd.read_csv(r"E:\MSPR_6.1_WildLens\infos_especes(2).csv")

# Sauvegarder en XLSX
df.to_excel(r"E:\MSPR_6.1_WildLens\infos_especes.xlsx", index=False, engine="openpyxl")

print("Conversion terminée : infos_especes.xlsx créé.")