import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


WAGI = "best.pt"
FOLDER_ZDJEC = "/Users/mateuszg/Documents/hackhaton_here/data/HACKATHON/Signs_images/"
PLIK_WYNIKOWY = "wyniki_detekcji.csv"

KAFEL = 640
OVERLAP = 0.1
CONF = 0.80
NMS_IOU = 0.45


MAX_DLUGI_BOK = 1600

PROG_WARIANCJI = 8.0

DEVICE = "cpu"
ZAPISUJ_CO_N = 50


def apply_nms(dets, iou_thresh):
    if len(dets) == 0:
        return dets
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(ovr <= iou_thresh)[0] + 1]
    return dets[keep]


def przetworz_zdjecie(model, sciezka):
    img = Image.open(sciezka).convert("RGB")
    W, H = img.size

    skala = 1.0
    if MAX_DLUGI_BOK and max(W, H) > MAX_DLUGI_BOK:
        skala = MAX_DLUGI_BOK / max(W, H)
        new_W, new_H = int(W * skala), int(H * skala)
        img = img.resize((new_W, new_H), Image.BILINEAR)
        W, H = new_W, new_H

    crops, offsets = [], []
    krok = int(KAFEL * (1 - OVERLAP))
    for y in range(0, H, krok):
        for x in range(0, W, krok):
            x2, y2 = min(x + KAFEL, W), min(y + KAFEL, H)
            if x2 <= x or y2 <= y:
                continue
            wycinek = img.crop((x, y, x2, y2))

            if PROG_WARIANCJI > 0:
                arr = np.asarray(wycinek)
                if arr.std() < PROG_WARIANCJI:
                    continue

            crops.append(wycinek)
            offsets.append((x, y))

    if not crops:
        return None

    raw = []
    for wycinek, (ox, oy) in zip(crops, offsets):
        res = model.predict(wycinek, imgsz=KAFEL, conf=CONF, device=DEVICE, verbose=False)[0]
        if not res.boxes:
            continue
        for b in res.boxes.data.cpu().numpy():
            raw.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, b[4], int(b[5])])

    if not raw:
        return None

    clean = apply_nms(np.array(raw), NMS_IOU)
    i = int(np.argmax(clean[:, 4]))
    nazwa = model.names[int(clean[i, 5])]
    pewnosc = float(clean[i, 4])
    return nazwa, pewnosc


def main():
    print(f"🚀 Ładuję model {WAGI} na {DEVICE}...")
    model = YOLO(WAGI)

    folder = Path(FOLDER_ZDJEC)
    zdjecia = sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.jpeg"))
    total = len(zdjecia)

    if total == 0:
        print("❌ Nie znaleziono żadnych zdjęć w podanym folderze!")
        return

    print(f"📂 Znaleziono plików: {total}")
    print(f"⚙️  KAFEL={KAFEL}, OVERLAP={OVERLAP}, MAX_DLUGI_BOK={MAX_DLUGI_BOK}, CONF={CONF}")
    print("Startuję detekcję...\n")

    wyniki = []
    wykrytych = 0
    t0 = time.time()

    pasek = tqdm(zdjecia, total=total, unit="img", dynamic_ncols=True)
    for licznik, p in enumerate(pasek, 1):
        try:
            det = przetworz_zdjecie(model, p)
        except Exception as e:
            pasek.write(f"⚠️ {p.name}: {e}")
            continue

        if det is not None:
            nazwa, pewnosc = det
            wykrytych += 1
            pasek.write(f"✅ {p.name} -> {nazwa} ({pewnosc:.2f})")
            wyniki.append({
                "file_name": p.name,
                "pelna_sciezka": str(p),
                "sign_type": nazwa,
                "confidence": round(pewnosc, 4),
            })

        sr_czas = (time.time() - t0) / licznik
        pasek.set_postfix(wykryte=wykrytych, sr=f"{sr_czas:.2f}s/img")

        if licznik % ZAPISUJ_CO_N == 0 and wyniki:
            pd.DataFrame(wyniki).to_csv(PLIK_WYNIKOWY, index=False)

    pasek.close()
    print(f"\n🎉 Koniec ({(time.time()-t0)/60:.1f} min). Wykryto znaki na {wykrytych}/{total} plikach.")

    if wyniki:
        pd.DataFrame(wyniki).to_csv(PLIK_WYNIKOWY, index=False)
        print(f"💾 Zapisano: {PLIK_WYNIKOWY}")
    else:
        print("⚠️ Brak detekcji — CSV nie został utworzony.")


if __name__ == "__main__":
    main()
