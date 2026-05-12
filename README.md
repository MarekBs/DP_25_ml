# Manuál – Tréning modelov behaviorálnej biometrie

## Požiadavky

Python 3.9 alebo novší.

```bash
pip install numpy pandas scipy scikit-learn xgboost joblib optuna
```

| Knižnica | Účel |
|---|---|
| `numpy` | numerické výpočty |
| `pandas` | načítanie a spracovanie CSV súborov |
| `scipy` | štatistické funkcie (skewness, kurtosis, FFT, find_peaks) |
| `scikit-learn` | ML algoritmy (SVM, Random Forest, KNN), pipeline, metriky |
| `xgboost` | XGBoost klasifikátor |
| `joblib` | ukladanie a načítanie `.pkl` modelov |
| `optuna` | len pre `find_optimal_params_optuna.py` |

---

## Štruktúra projektu

```
data/
├── swipe/           – záznamy swipe gesta
├── zoom/            – záznamy zoom/pinch gesta
├── walk/            – záznamy chôdze
├── stol/            – záznamy položenia na stôl
└── zdvihnutie/      – záznamy zdvihnutia k uchu

ml_training/
├── train_swipe.py              – tréning: swipe gesto
├── train_zoom.py               – tréning: zoom/pinch gesto
├── train_walk.py               – tréning: chôdza
├── train_table.py              – tréning: položenie na stôl
├── train_pickup.py             – tréning: zdvihnutie k uchu
├── training.py                 – zdieľaná logika trénovania a vyhodnocovania
├── feature_selection.py        – korelačný filter a RF importance filter
├── find_optimal_k.py           – prehľad presnosti pre rôzne počty príznakov
├── find_optimal_threshold.py   – hľadanie optimálnych prahov
├── find_optimal_params_optuna.py – hľadanie hyperparametrov (Optuna)
└── multimodal/
      ├── core.py               – zdieľané funkcie pre multimodálne skripty
      ├── train_swipe.py        – tréning: swipe gesto (dotyk + senzory)
      └── train_zoom.py         – tréning: zoom/pinch gesto (dotyk + senzory)

gesture_model_zdvihnutie.pkl    – natrénovaný model (Random Forest) pre autentifikáciu
optimal_thresholds.json         – optimálne prahy pre všetky modely
server.py                       – autentifikačný server
requirements_server.txt         – závislosti servera
```

Výstupné `.pkl` súbory sa uložia do priečinka, z ktorého sa skript spúšťa.

> **Poznámka:** V repozitári je priložený natrénovaný model `gesture_model_zdvihnutie.pkl` (Random Forest), ktorý sa používa na autentifikáciu. Optimálne prahy pre tento aj ostatné modely sú v `optimal_thresholds.json`.

---

##  Základný tréning

Použijú sa **všetky príznaky** + **optimálne hyperparametre** uložené v `training.py` (funkcia `make_models()`). Hyperparametre sú už nastavené – stačí spustiť:

```bash
cd ml_training
python train_<gesto>.py
```

Dostupné gestá: `swipe`, `zoom`, `walk`, `table`, `pickup`.

Ak chceš zmeniť hyperparametre modelov (napr. po novom Optuna behu, alebo na pôvodné základné), uprav hodnoty priamo v `training.py` a znovu spusti tréningový skript.

Multimodálne varianty (dotyk + senzory) sú dostupné iba pre `swipe` a `zoom`:

```bash
cd ml_training/multimodal
python train_swipe.py
python train_zoom.py
```

---

##  Tréning s top-K príznakmi

Použije sa **top-K najdôležitejších príznakov** + **optimálne hyperparametre** z `training.py`. Hodnoty K pre kombináciu gesto/model sú uložené v `feature_selection.py` v slovníku `OPTIMAL_K`.

| Prepínač | Popis |
|---|---|
| `--fs corr` | rýchly korelačný filter (odstraňuje redundantné príznaky) |
| `--fs full` | korelačný filter + RF importance (top K príznakov pre daný model) |

Voliteľne je možné určiť model, pre ktorý sa vyberie optimálny počet príznakov (predvolený: `Random Forest`):

```
--model SVM
--model "Random Forest"
--model XGBoost
--model KNN
```

**Príklady:**
```bash
python train_swipe.py  --fs full --model "Random Forest"
python train_zoom.py   --fs full --model XGBoost
python train_table.py  --fs corr
python train_pickup.py --fs full --model SVM
```

Funguje iba pre základné tréningové skripty (nie multimodal).

---

## Top-K + optimálny prah (kompletné vyhodnotenie)

Najkomplexnejší experiment – kombinuje **top-K príznaky** + **optimálne hyperparametre** (z `training.py`) + **optimálny prah** (cez OOF ROC krivku). Pre každé gesto a model vypíše tabuľku metrík (`Prah`, `Acc`, `FAR`, `FRR`, `EER`, `Prec`, `Rec`, `F1`, `AUC`, `CV-AUC`, `Hits`, `Miss`).

```bash
python find_optimal_threshold.py
```

Výstup: tabuľka v konzole + `optimal_thresholds.json` (používa ho aplikácia pri autentifikácii).

---

## Pomocné skripty (hľadanie nových optimálnych hodnôt)

Tieto skripty sa spúšťajú iba ak chceš znovu nájsť optimálne hodnoty K alebo hyperparametrov. Existujúce hodnoty sú už nastavené v `feature_selection.py` (`OPTIMAL_K`) a `training.py` (`make_models()`).

**Nové optimálne K pre dané gesto** – vypíše presnosť modelov pre rôzne počty príznakov (k = 2, 4, ..., všetky). Nájdené hodnoty zapíš do `OPTIMAL_K`:
```bash
python find_optimal_k.py <gesto>
```

**Nové optimálne hyperparametre (Optuna)** – trvá dlho. Nájdené hodnoty zapíš ručne do `training.py`:
```bash
python find_optimal_params_optuna.py
```

---

## Ďalšie parametre

Stiahnutie dát z Firebase Storage (vyžaduje `serviceAccountKey.json`):
```bash
python train_<gesto>.py --download
```

Vlastná cesta k dátam (ak dáta nie sú v predvolenom priečinku):
```bash
python train_<gesto>.py --data-dir <cesta/k/datam>
```

Veľkosť kĺzavého okna pre chôdzu (predvolene: okno = 256, krok = 128):
```bash
python train_walk.py --window-size <velkost> --window-step <krok>
```

---

## Prehľad výstupných súborov

| Skript | Výstupný súbor |
|---|---|
| `train_swipe.py` | `swipe_model.pkl` |
| `train_zoom.py` | `zoom_model.pkl` |
| `train_walk.py` | `walk_model.pkl` |
| `train_table.py` | `gesture_model_stol.pkl` |
| `train_pickup.py` | `gesture_model_zdvihnutie.pkl` |
| `multimodal/train_swipe.py` | `swipe_model_mm.pkl` |
| `multimodal/train_zoom.py` | `zoom_model_mm.pkl` |
| `find_optimal_threshold.py` | `optimal_thresholds.json` |

---

**1. Natrénovať model pre gesto zdvihnutia k uchu:**
```bash
cd ml_training
python train_pickup.py --fs full --model SVM
```
Výstup: `gesture_model_zdvihnutie.pkl`

## Server pre autentifikáciu

**2. Spustiť server**:
```bash
python server.py
```

Server beží na adrese `http://0.0.0.0:5000`. Server a zariadenie s aplikáciou musia byť na **rovnakej sieti**. IP adresu zariadenia so serverom je potrebné nastaviť v zdrojovom kóde aplikácie (`VerifyPickupFragment.kt` – konštanta `SERVER_URL`).


## Ako funguje tréning

Každý skript vyhodnotí 4 algoritmy: SVM, Random Forest, XGBoost, KNN. Pre každého používateľa sa natrénuje samostatný model (prístup 1-vs-all) – model sa učí rozoznať "toto som ja" od "toto nie som ja".

Po vyhodnotení sa vyberie algoritmus s najlepšou priemernou presnosťou a jeho modely sa uložia do `.pkl` súboru:

```json
{
  "models":        { "user_A": "Pipeline", "user_B": "Pipeline" },
  "feature_names": [ "x_mean", "y_std", "..." ],
  "model_type":    "Random Forest"
}
```

Pri autentifikácii aplikácia načíta model daného používateľa, zavolá `predict_proba()` a výsledok porovná s prahom z `optimal_thresholds.json`.

