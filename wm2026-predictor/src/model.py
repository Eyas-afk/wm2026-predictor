"""
Trainiert ein Modell, das aus ELO-Differenz + Kontext-Features die
Wahrscheinlichkeit für Heimsieg / Unentschieden / Auswärtssieg vorhersagt.

Modell: Gradient Boosting Classifier (scikit-learn) mit 3 Klassen.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Projekt-Wurzelverzeichnis ermitteln (zwei Ebenen über diesem Skript,
# src/model.py -> src/ -> Projekt-Root). So funktioniert das Skript auf
# jedem Rechner, unabhängig davon wo das Projekt liegt - keine
# hartcodierten Pfade nötig.
PROJECT_ROOT = Path(__file__).parent.parent

# Mapping von Ergebnis-Buchstabe zu Zahl, weil scikit-learn nur numerische
# Zielvariablen akzeptiert. Buchstaben bleiben zusätzlich als "result"-Spalte
# erhalten (siehe build_features), damit die Daten beim manuellen Anschauen
# lesbar bleiben statt nur 0/1/2 zu zeigen.
RESULT_MAP = {"H": 0, "D": 1, "A": 2}  # Heimsieg, Draw, Auswärtssieg


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Wandelt die rohe ELO-Historie (aus elo.py) in ein Format um, das das
    Modell trainieren kann: numerische Features + numerische Zielvariable.
    """
    # .copy(), damit wir das übergebene history-DataFrame nicht versehentlich
    # verändern (defensive Programmierung - vermeidet Seiteneffekte,
    # falls history später im Aufrufer-Code noch gebraucht wird).
    df = history.copy()

    # Wichtigstes Feature: die ELO-Differenz. Wird explizit vorberechnet statt
    # dem Modell nur die zwei Rohwerte zu geben - erspart dem Modell, diesen
    # Zusammenhang erst selbst "entdecken" zu müssen (Feature Engineering).
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]

    # "neutral" kommt als Text-Wahrheitswert aus der CSV (z.B. "TRUE"/"FALSE",
    # ggf. uneinheitlich geschrieben) - Modelle brauchen aber echte Zahlen.
    # str.upper() fängt Schreibweise-Varianten ab, .eq("TRUE") vergleicht,
    # astype(int) macht daraus 1/0.
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)

    # Zielvariable aus den Torergebnissen ableiten.
    def result(row):
        if row["home_score"] > row["away_score"]:
            return "H"
        elif row["home_score"] == row["away_score"]:
            return "D"
        else:
            return "A"

    df["result"] = df.apply(result, axis=1)
    df["target"] = df["result"].map(RESULT_MAP)

    return df


def main():
    history = pd.read_csv(PROJECT_ROOT / "data" / "elo_history.csv")
    df = build_features(history)

    # home_elo_pre/away_elo_pre bleiben zusätzlich zu elo_diff im Feature-Set,
    # falls die absolute Stärke (nicht nur der Unterschied) zusätzliche
    # Information liefert (z.B. zwei sehr starke Teams könnten sich anders
    # verhalten als zwei sehr schwache Teams mit demselben elo_diff).
    features = ["elo_diff", "home_elo_pre", "away_elo_pre", "neutral"]
    X = df[features]
    y = df["target"]

    # 85/15-Split. random_state=42 macht den Split reproduzierbar (gleicher
    # Split bei jedem Lauf -> fairer Vergleich, wenn wir später Modell-
    # Änderungen testen). stratify=y erhält die Klassenverteilung (v.a.
    # wichtig, weil Unentschieden ohnehin die seltenste Klasse ist - ein
    # unglücklicher Zufalls-Split könnte das Ungleichgewicht verschärfen).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Gradient Boosting: viele schwache, flache Bäume (max_depth=3) statt
    # wenige tiefe - reduziert Overfitting-Risiko. Kleine learning_rate (0.05)
    # + entsprechend viele Bäume (n_estimators=300), damit sich die kleinen
    # Korrekturschritte zu einem starken Gesamtmodell aufsummieren, ohne dass
    # einzelne Bäume zu stark auf Trainings-Ausreißer reagieren. subsample=0.8
    # gibt jedem Baum eine andere zufällige 80%-Stichprobe der Trainingsdaten,
    # was die Bäume diverser macht und zusätzlich vor Overfitting schützt.
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # predict() liefert die wahrscheinlichste Klasse (0/1/2), predict_proba()
    # liefert alle drei Wahrscheinlichkeiten. Wir brauchen beides: predict()
    # für die Accuracy-Metrik, predict_proba() für Log Loss UND für die
    # spätere Monte-Carlo-Simulation in simulate.py (die braucht echte
    # Wahrscheinlichkeiten zum Würfeln, keine feste Ja/Nein-Antwort).
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("=== Modell-Evaluation (Testset) ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    # Log Loss bestraft übertrieben selbstsichere Fehlvorhersagen deutlich
    # härter als vorsichtige Fehlvorhersagen - wichtiger als Accuracy, weil
    # es die Qualität der Wahrscheinlichkeiten selbst bewertet, nicht nur
    # ob die wahrscheinlichste Klasse zufällig stimmte.
    print(f"Log Loss: {log_loss(y_test, probs):.3f}")
    print()
    print(classification_report(y_test, preds, target_names=["Heimsieg", "Unentschieden", "Auswärtssieg"]))

    # Baseline-Vergleich: ein Modell, das immer stur "Heimsieg" tippt (nutzt
    # nur den bekannten generellen Heimvorteil im Fußball, ohne jede
    # Team-spezifische Information). Zeigt, ob der ganze ELO+ML-Aufwand
    # tatsächlich einen Mehrwert bringt, oder ob wir mit einer trivialen
    # Daumenregel fast genauso gut abgeschnitten hätten.
    baseline_preds = np.zeros_like(y_test)
    print(f"Baseline (immer Heimsieg) Accuracy: {accuracy_score(y_test, baseline_preds):.3f}")

    # Zeigt, welche Features das Modell tatsächlich am meisten nutzt -
    # praktischer Sanity-Check: elo_diff sollte dominieren, wenn unser
    # Feature Engineering (siehe build_features) sinnvoll war.
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:15s} {imp:.3f}")

    # Modell auf Disk speichern, damit simulate.py es laden kann, ohne
    # jedes Mal neu zu trainieren.
    joblib.dump(model, PROJECT_ROOT / "data" / "match_model.pkl")
    print("\n✅ Modell gespeichert: data/match_model.pkl")


if __name__ == "__main__":
    main()