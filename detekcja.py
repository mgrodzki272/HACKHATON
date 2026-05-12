import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ==========================================
# Tutaj wpisz surowe nazwy klas ze swojego modelu
# i to, na co mają zostać zamienione:
MAPOWANIE_ZNAKOW = {
    "speed_limit_30": "30",
    "speed_limit_50": "50",
    "speed_limit_70": "70",
    "stop": "stop",
    # itd... dopisz swoje klasy z YOLO
}


# ==========================================

def oblicz_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    czesc_wspolna = max(0.0, xB - xA) * max(0.0, yB - yA)
    poleA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    poleB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return czesc_wspolna / float(poleA + poleB - czesc_wspolna + 1e-9)


def usun_duplikaty_nms(boxes, scores, threshold):
    kolejnosc = scores.argsort()[::-1]
    zachowane_indeksy = []

    while kolejnosc.size > 0:
        i = kolejnosc[0]
        zachowane_indeksy.append(i)

        reszta = kolejnosc[1:]
        if reszta.size == 0:
            break

        iou_z_reszta = np.array([oblicz_iou(boxes[i], boxes[j]) for j in reszta])
        kolejnosc = reszta[iou_z_reszta < threshold]

    return zachowane_indeksy


def skanuj_obraz(model, img, prog_pewnosci, kafel, overlap):
    szerokosc, wysokosc = img.size
    krok = int(kafel * (1 - overlap))
    znalezione_obiekty = []

    for y in range(0, wysokosc, krok):
        for x in range(0, szerokosc, krok):
            x_koniec = min(x + kafel, szerokosc)
            y_koniec = min(y + kafel, wysokosc)

            if x_koniec <= x or y_koniec <= y:
                continue

            wycinek = img.crop((x, y, x_koniec, y_koniec))
            wyniki = model(wycinek, imgsz=kafel, conf=prog_pewnosci, verbose=False)[0]

            if not wyniki.boxes:
                continue

            dane_boxow = wyniki.boxes.data.cpu().numpy()

            for obiekt in dane_boxow:
                x1, y1, x2, y2, pewnosc, klasa = obiekt
                znalezione_obiekty.append([
                    x1 + x, y1 + y, x2 + x, y2 + y,
                    pewnosc, int(klasa)
                ])

    return znalezione_obiekty


def wykryj_znaki(sciezka_zdjecia, sciezka_modelu="best.pt", kafel=640, overlap=0.25, nms_iou=0.45, conf_standard=0.85,
                 conf_ratunkowy=0.01):
    nazwa_pliku = os.path.basename(sciezka_zdjecia)

    model = YOLO(sciezka_modelu)
    img = Image.open(sciezka_zdjecia).convert("RGB")

    wszystkie_detekcje = skanuj_obraz(model, img, prog_pewnosci=conf_standard, kafel=kafel, overlap=overlap)

    if not wszystkie_detekcje:
        wszystkie_detekcje = skanuj_obraz(model, img, prog_pewnosci=conf_ratunkowy, kafel=kafel, overlap=overlap)

    # Jeśli pomimo trybu ratunkowego jest brak detekcji -> zwracamy "null"
    if not wszystkie_detekcje:
        return nazwa_pliku, "null"

    detekcje_np = np.array(wszystkie_detekcje)
    boxy = detekcje_np[:, :4]
    pewnosci = detekcje_np[:, 4]
    klasy = detekcje_np[:, 5].astype(int)

    finalne_indeksy = []
    for unikalna_klasa in np.unique(klasy):
        indeksy_klasy = np.where(klasy == unikalna_klasa)[0]
        boxy_klasy = boxy[indeksy_klasy]
        pewnosci_klasy = pewnosci[indeksy_klasy]

        zachowane = usun_duplikaty_nms(boxy_klasy, pewnosci_klasy, nms_iou)
        finalne_indeksy.extend(indeksy_klasy[zachowane])

    nazwy_klas = model.names
    wyniki_koncowe = []

    for i in finalne_indeksy:
        nazwa_z_modelu = nazwy_klas.get(klasy[i], str(klasy[i]))

        # Podmieniamy nazwę na podstawie słownika.
        # Jeśli nie ma jej w słowniku, zostawia oryginalną nazwę z YOLO.
        nazwa_docelowa = MAPOWANIE_ZNAKOW.get(nazwa_z_modelu, nazwa_z_modelu)

        pewnosc = float(pewnosci[i])
        wyniki_koncowe.append((nazwa_docelowa, pewnosc))


    if not wyniki_koncowe:
        return nazwa_pliku, "null"

    return nazwa_pliku, wyniki_koncowe



if __name__ == "__main__":
    sciezka = "/Users/mateuszg/Documents/hackhaton_here/data/HACKATHON/Signs_images/HT412_1765878407/Front/000000-1449913713914751-PERSPECTIVE_FRONT.jpg"

    nazwa_pliku, wyniki = wykryj_znaki(sciezka, "best.pt")

    print(f"Plik: {nazwa_pliku}")

    if wyniki == "null":
        print("Wynik: null")
    else:
        print("Wykryte znaki:")
        for znak, pewnosc in wyniki:
            print(f"- {znak} (Pewność: {pewnosc:.4f})")