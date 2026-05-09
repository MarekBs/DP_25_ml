# Behaviorálna biometria – ML tréning a server

Tréningové skripty, dáta a Flask server pre systém autentifikácie používateľa na základe behaviorálnych biometrík.  
Android aplikácia sa nachádza v repozitári [DP_25](https://github.com/MarekBs/DP_25).

---

## Štruktúra projektu

```
├── ml_training/                   # Tréningové skripty
│   ├── train_swipe.py             # swipe gesto (dotyk)
│   ├── train_zoom.py              # zoom/pinch gesto (dotyk)
│   ├── train_walk.py              # chôdza (akcelerometer)
│   ├── train_table.py             # položenie na stôl (acc + gyroskop)
│   ├── train_pickup.py            # zdvihnutie k uchu (acc + gyroskop)
│   ├── training.py                # zdieľaná logika trénovania
│   ├── feature_selection.py       # výber príznakov
│   ├── find_optimal_k.py          # hľadanie optimálneho počtu príznakov
│   ├── find_optimal_threshold.py  # hľadanie optimálneho prahu
│   ├── find_optimal_params_optuna.py # optimalizácia hyperparametrov
│   └── multimodal/                # dotyk + senzory
│       ├── core.py
│       ├── train_swipe.py
│       └── train_zoom.py
├── data/                          # Tréningové dáta
│   ├── swipe/
│   ├── zoom/
│   ├── walk/
│   ├── stol/
│   └── zdvihnutie/
├── server.py                      # Flask server pre live verifikáciu
├── requirements_server.txt        # Závislosti servera
├── optimal_thresholds.json        # Prahy pre autentifikáciu
└── Manual.txt                     # Podrobný manuál
```

---

## Požiadavky

- Python 3.9+

### Inštalácia závislostí pre tréning

```bash
pip install numpy pandas scipy scikit-learn xgboost joblib optuna
```

### Inštalácia závislostí pre server

```bash
pip install -r requirements_server.txt
```

---

## Tréning modelov

```bash
cd ml_training

python train_swipe.py
python train_zoom.py
python train_walk.py
python train_table.py
python train_pickup.py
```

Podrobný návod na spustenie experimentov nájdeš v [Manual.txt](Manual.txt).

---

## Spustenie servera

```bash
python server.py
```

Server beží na porte `5000` a poskytuje API pre live verifikáciu gest z Android aplikácie.

---

## Algoritmy

- Random Forest
- SVM
- KNN
- XGBoost
- optimalizácia hyperparametrov pomocou Optuna
