import pandas as pd
import os

# 1. Wczytaj tabelę z metadanymi (tą ze zdjęcia z lat/lon)
df = pd.read_csv("metadata_images.csv")


results = []

print("🚀 Rozpoczynam masowe przetwarzanie zdjęć...")

for index, row in df.iterrows():

    image_path = os.path.join("/Users/mateuszg/Documents/hackhaton_here/data/...", row['image_id'] + ".jpg")

    if os.path.exists(image_path):

        detekcje = glowna_funkcja_detekcji(image_path)

        if detekcje:
            for d in detekcje:

                results.append({
                    "image_id": row['image_id'],
                    "lat": row['latitude'],
                    "lon": row['longitude'],
                    "detected_sign": d['znak'],
                    "confidence": d['pewnosc']
                })
        else:

            results.append({
                "image_id": row['image_id'],
                "lat": row['latitude'],
                "lon": row['longitude'],
                "detected_sign": "None",
                "confidence": 0
            })


final_df = pd.DataFrame(results)
final_df.to_csv("final_results_for_map.csv", index=False)
print("✅ Przetwarzanie zakończone. Dane gotowe do wrzucenia na mapę!")