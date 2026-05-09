import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import joblib

N_FOLDS = 5


def make_models(params=None):
    p = params or {}
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42,
                        **{"kernel": "rbf", "C": 1.9911, "gamma": "auto", **p.get("SVM", {})}))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1,
                        **{"n_estimators": 100, "max_depth": 30, "min_samples_leaf": 1,
                           "min_samples_split": 2, "max_features": "sqrt", "bootstrap": True,
                           **p.get("Random Forest", {})}))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1,
                        **{"n_estimators": 150, "max_depth": 4, "learning_rate": 0.0984,
                           "subsample": 0.9629, "colsample_bytree": 0.9769,
                           "min_child_weight": 1, "reg_alpha": 0.000285, "reg_lambda": 5.2905,
                           **p.get("XGBoost", {})}))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(**{"n_neighbors": 10, "metric": "manhattan",
                           "weights": "distance", "algorithm": "brute", **p.get("KNN", {})}))
        ]),
    }


def train_and_evaluate(X, y, feature_names, output_pkl=None, min_samples=2, params=None, only_models=None, verbose=True):
    users       = np.unique(y)
    model_names = [k for k in make_models(params).keys()
                   if only_models is None or k in only_models]
    results     = {name: {"fars": [], "frrs": [], "eers": [], "aucs": [], "accs": [],
                          "precs": [], "recs": [], "f1s": [], "hits": [], "misses": [],
                          "cv_aucs": []}
                   for name in model_names}
    best_models = {name: {} for name in model_names}

    for target_user in users:
        y_bin = (y == target_user).astype(int)

        rng     = np.random.default_rng(42)
        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        neg_idx = neg_idx[:len(pos_idx)]

        if len(pos_idx) < min_samples:
            if verbose:
                print(f"  [SKIP] {target_user}: príliš málo pozitívnych vzoriek")
            continue

        n_pos_test = max(1, int(round(len(pos_idx) * 0.30)))
        n_neg_test = max(1, int(round(len(neg_idx) * 0.30)))

        test_idx     = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
        trainval_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])

        X_trainval, y_trainval = X[trainval_idx], y_bin[trainval_idx]
        X_test,     y_test     = X[test_idx],     y_bin[test_idx]

        if len(np.unique(y_trainval)) < 2:
            continue

        n_splits = min(N_FOLDS, int(np.min(np.bincount(y_trainval))))
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for name, model in {k: v for k, v in make_models(params).items() if k in model_names}.items():
            if name == "KNN":
                clf = model.named_steps["clf"]
                min_cv_size = len(X_trainval) * (n_splits - 1) // n_splits
                clf.n_neighbors = min(clf.n_neighbors, min_cv_size - 1)
            cv_auc = cross_val_score(model, X_trainval, y_trainval,
                                     cv=cv, scoring="roc_auc").mean()
            model.fit(X_trainval, y_trainval)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            TP = int(((y_pred == 1) & (y_test == 1)).sum())
            FP = int(((y_pred == 1) & (y_test == 0)).sum())
            TN = int(((y_pred == 0) & (y_test == 0)).sum())
            FN = int(((y_pred == 0) & (y_test == 1)).sum())

            FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

            if len(np.unique(y_test)) > 1:
                fpr_c, tpr_c, _ = roc_curve(y_test, y_proba)
                fnr_c = 1 - tpr_c
                eer = float((fpr_c + fnr_c)[np.argmin(np.abs(fpr_c - fnr_c))]) / 2
                auc = roc_auc_score(y_test, y_proba)
            else:
                eer = auc = 0.0

            results[name]["cv_aucs"].append(cv_auc)
            results[name]["fars"].append(FAR)
            results[name]["frrs"].append(FRR)
            results[name]["eers"].append(eer)
            results[name]["aucs"].append(auc)
            results[name]["accs"].append(accuracy_score(y_test, y_pred))
            results[name]["precs"].append(precision_score(y_test, y_pred, zero_division=0))
            results[name]["recs"].append(recall_score(y_test, y_pred, zero_division=0))
            results[name]["f1s"].append(f1_score(y_test, y_pred, zero_division=0))
            results[name]["hits"].append(int((y_pred == y_test).sum()))
            results[name]["misses"].append(int((y_pred != y_test).sum()))
            best_models[name][target_user] = model

    if verbose:
        hdr = f"\n{'Model':<20} {'Acc':>6} {'FAR':>6} {'FRR':>6} {'EER':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV-AUC':>8} {'Hits':>8} {'Miss':>8}"
        print(hdr)
        print("-" * len(hdr))
        for name in model_names:
            r = results[name]
            if not r["accs"]:
                continue
            print(f"{name:<20} {np.mean(r['accs']):>6.3f} {np.mean(r['fars']):>6.3f} "
                  f"{np.mean(r['frrs']):>6.3f} {np.mean(r['eers']):>6.3f} {np.mean(r['precs']):>6.3f} "
                  f"{np.mean(r['recs']):>6.3f} {np.mean(r['f1s']):>6.3f} {np.mean(r['aucs']):>6.3f} "
                  f"{np.mean(r['cv_aucs']):>8.3f} {sum(r['hits']):>8} {sum(r['misses']):>8}")

    best_name = max(model_names, key=lambda k: np.mean(results[k]["accs"]) if results[k]["accs"] else 0)
    if output_pkl is not None:
        # retrain each user's model on ALL data (not just the 70 % trainval split)
        final_models = {}
        for target_user in users:
            y_bin = (y == target_user).astype(int)
            rng = np.random.default_rng(42)
            pos_idx = np.where(y_bin == 1)[0]
            neg_idx = np.where(y_bin == 0)[0]
            rng.shuffle(neg_idx)
            neg_idx = neg_idx[:len(pos_idx)]
            all_idx = np.concatenate([pos_idx, neg_idx])
            X_all, y_all = X[all_idx], y_bin[all_idx]
            if len(np.unique(y_all)) < 2 or len(y_all) < min_samples:
                continue
            final_model = make_models(params)[best_name]
            if best_name == "KNN":
                final_model.named_steps["clf"].n_neighbors = min(
                    final_model.named_steps["clf"].n_neighbors, len(X_all) - 1
                )
            final_model.fit(X_all, y_all)
            final_models[target_user] = final_model

        joblib.dump({
            "models": final_models,
            "feature_names": feature_names,
            "model_type": best_name,
        }, output_pkl)
        print(f"\nNajlepší model: {best_name} "
              f"(avg acc={np.mean(results[best_name]['accs']):.4f}) -> {output_pkl}"
              f"\n  Retrain na 100 % dát: {len(final_models)} používateľov")
    return results