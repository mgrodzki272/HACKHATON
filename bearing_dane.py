import json
import math
import pandas as pd
import numpy as np

# ── PARAMETRY ─────────────────────────────────────────────────────────────────
GEOJSON_ZNAKI = "Bulgaria_spdlmt_signs.geojson"
CSV_DETEKCJE = "gotowedane_z_metadanymi.csv"
PLIK_WYNIKOWY = "confidence_scores.csv"

MAX_DYSTANS_M = 45  # bufor wokół znaku
MAX_BEARING_DIFF = 60  # tolerancja kąta kamery vs znaku


# ── HELPERS ───────────────────────────────────────────────────────────────────
def dystans_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    a = math.radians(lat2 - lat1)
    b = math.radians(lon2 - lon1)
    c = math.sin(a / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(b / 2) ** 2
    return R * 2 * math.asin(math.sqrt(c))


def kat_diff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


# ── WCZYTAJ ───────────────────────────────────────────────────────────────────
print("Wczytuję dane...")
with open(GEOJSON_ZNAKI, encoding="utf-8") as f:
    znaki = [{"node_id": x["properties"]["node_id"],
              "speed_lmt": int(x["properties"]["speed_lmt"]),
              "bearing": float(x["properties"]["bearing"]),
              "lat": float(x["properties"]["lat"]),
              "lon": float(x["properties"]["lon"])}
             for x in json.load(f)["features"]]

det = pd.read_csv(CSV_DETEKCJE)
print(f"  Znaków HERE: {len(znaki)}, zdjęć: {len(det)}")

# ── SCORE PER ZNAK ────────────────────────────────────────────────────────────
print("Liczę confidence scores...")
wyniki = []

for znak in znaki:
    # Zdjęcia w buforze + pasujący bearing
    pasujace = det[det.apply(lambda r:
                             dystans_m(r["lat"], r["lon"], znak["lat"], znak["lon"]) < MAX_DYSTANS_M and
                             kat_diff(r["front_bearing"], znak["bearing"]) < MAX_BEARING_DIFF,
                             axis=1)].copy()

    n_zdjec = len(pasujace)
    n_wykrytych = pasujace["sign_type"].notna().sum()

    if n_zdjec == 0:
        score = 0.0
        avg_conf = None
    else:
        # detection_rate: % zdjęć które cokolwiek wykryły
        detection_rate = n_wykrytych / n_zdjec

        # match_rate: % zdjęć gdzie wykryty limit == limit HERE
        n_match = (pasujace["sign_type"] == znak["speed_lmt"]).sum()
        match_rate = n_match / n_zdjec

        # średnia pewność YOLO (tylko dla wykrytych)
        avg_conf = pasujace.loc[pasujace["sign_type"].notna(), "confidence"].mean()
        avg_conf = float(avg_conf) if not np.isnan(avg_conf) else 0.0

        # Ważony score: detekcja 50%, zgodność limitu 40%, pewność YOLO 10%
        score = round(0.50 * detection_rate + 0.40 * match_rate + 0.10 * avg_conf, 3)

    wyniki.append({
        "node_id": znak["node_id"],
        "speed_lmt": znak["speed_lmt"],
        "lat": znak["lat"],
        "lon": znak["lon"],
        "n_zdjec": n_zdjec,
        "n_wykrytych": int(n_wykrytych),
        "avg_yolo_conf": round(avg_conf, 3) if avg_conf is not None else None,
        "existence_confidence_score": score,
    })

# ── ZAPIS ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(wyniki).sort_values("existence_confidence_score", ascending=False)
df.to_csv(PLIK_WYNIKOWY, index=False)

print(f"\n✅ Zapisano: {PLIK_WYNIKOWY}")
print(f"   Znaków z pokryciem SLI:    {(df['n_zdjec'] > 0).sum()} / {len(df)}")
print(f"   Score > 0.7 (istnieje):    {(df['existence_confidence_score'] > 0.7).sum()}")
print(f"   Score < 0.3 (brak/pewne):  {(df['existence_confidence_score'] < 0.3).sum()}")
print(f"\nTop 5:")
print(
    df[["node_id", "speed_lmt", "n_zdjec", "n_wykrytych", "existence_confidence_score"]].head(5).to_string(index=False))
