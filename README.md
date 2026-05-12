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
```

Výstupné `.pkl` súbory sa uložia do priečinka, z ktorého sa skript spúšťa.

---

## 1. Základný tréning (predvolené nastavenia)

Skripty sa spúšťajú z adresára `ml_training/`. Každé gesto má vlastný tréningový skript:

```bash
cd ml_training
python train_<gesto>.py
```

Dostupné gestá: `swipe`, `zoom`, `walk`, `table`, `pickup`

Použijú sa **všetky príznaky** a **optimálne hyperparametre uložené v `training.py`** (funkcia `make_models()`). Ak chceš zmeniť hyperparametre modelov (napr. po novom Optuna behu, alebo pôvodné základné), uprav hodnoty priamo v `training.py` a znovu spusti tréningový skript.

Multimodálne varianty (dotyk + senzory) sú dostupné iba pre `swipe` a `zoom`:

```bash
cd ml_training/multimodal
python train_swipe.py
python train_zoom.py
```

---

## 2. Tréning s top-K príznakmi

Funguje iba pre základné tréningové skripty (nie multimodal). Predvolene je feature selection vypnutá – použijú sa všetky príznaky.

| Prepínač | Popis |
|---|---|
| `--fs corr` | rýchly korelačný filter (odstraňuje redundantné príznaky) |
| `--fs full` | korelačný filter + RF importance (top K príznakov pre daný model) |

Hodnoty K pre kombináciu gesto/model sú uložené v `feature_selection.py` v slovníku `OPTIMAL_K`.

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

---

## 3. Hľadanie optimálnych hyperparametrov (Optuna)

Hľadá optimálne hyperparametre modelov pomocou Optuny. Spúšťa sa iba pri hľadaní nových hyperparametrov – trvá dlho. Nájdené hodnoty treba následne ručne preniesť do `training.py` (funkcia `make_models()`).

```bash
python find_optimal_params_optuna.py
```

---

## 4. Hľadanie optimálneho prahu

Po tréningu nájde optimálny prah rozhodovania pre každú kombináciu gesta a modelu (cez OOF ROC krivku, používa top-K príznakov z `OPTIMAL_K`).

```bash
python find_optimal_threshold.py
```

Výstup: `optimal_thresholds.json`

---

## 5. Hľadanie optimálneho K

Zobrazí presnosť modelov pre rôzne počty príznakov (k = 2, 4, ..., všetky). Pomáha určiť optimálne K pre dané gesto, ktoré sa potom zapíše do `OPTIMAL_K` v `feature_selection.py`.

```bash
python find_optimal_k.py swipe
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

