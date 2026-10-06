import itertools
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


SEED = 42
DATASET_PATH = "dataset_ataques_cajeros.csv"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo del dataset: {path}")
    df = pd.read_csv(path)
    if "clase" not in df.columns:
        raise ValueError('El dataset debe contener la columna objetivo "clase".')
    return df


def prepare_data(df: pd.DataFrame):
    X = df.drop("clase", axis=1)
    y = df["clase"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def crear_modelo(input_dim: int, neuronas=(32, 16, 8, 4), learning_rate: float = 0.001) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(neuronas[0], activation="relu"),
            Dense(neuronas[1], activation="relu"),
            Dense(neuronas[2], activation="relu"),
            Dense(neuronas[3], activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def search_best_model(X_train, y_train, X_test, y_test):
    arquitecturas = [
        (16, 8, 4, 2),
        (32, 16, 8, 4),
        (64, 32, 16, 8),
    ]
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [4, 8, 16]

    resultados = []
    best = {
        "accuracy": -1.0,
        "model": None,
        "history": None,
        "config": None,
    }

    early = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-5,
        verbose=0,
    )

    for arquitectura, lr, batch in itertools.product(arquitecturas, learning_rates, batch_sizes):
        print("=" * 70)
        print(f"Arquitectura: {arquitectura}")
        print(f"Learning Rate: {lr}")
        print(f"Batch Size: {batch}")

        model = crear_modelo(X_train.shape[1], arquitectura, lr)
        history = model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=batch,
            validation_split=0.20,
            callbacks=[early, reduce_lr],
            verbose=0,
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        resultados.append(
            {
                "arquitectura": str(arquitectura),
                "learning_rate": lr,
                "batch_size": batch,
                "loss": float(loss),
                "accuracy": float(acc),
            }
        )
        print(f"Accuracy: {acc * 100:.2f}%")

        if acc > best["accuracy"]:
            best.update(
                {
                    "accuracy": float(acc),
                    "model": model,
                    "history": history,
                    "config": (arquitectura, lr, batch),
                }
            )

    resultados_df = pd.DataFrame(resultados).sort_values("accuracy", ascending=False)
    resultados_df.to_csv("resultados_modelos.csv", index=False)
    return best["model"], best["history"], best["config"], best["accuracy"], resultados_df


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_acc = 0.0

    for t in np.arange(0.20, 0.81, 0.01):
        preds = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(t)

    return best_threshold, best_acc


def save_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Ataque"],
        yticklabels=["Normal", "Ataque"],
    )
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()
    return roc_auc


def plot_loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title("Curva de pérdida")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()


def main():
    set_seeds()
    print("Directorio actual:", os.getcwd())

    df = load_dataset(DATASET_PATH)
    print(df.head())
    print("\nInformación del dataset")
    print(df.info())
    print("\nPrimeras filas")
    print(df.head())
    print("\nDistribución de clases")
    print(df["clase"].value_counts())

    X_train, X_test, y_train, y_test, _ = prepare_data(df)
    print("\nTamaño del conjunto de entrenamiento:", X_train.shape)
    print("Tamaño del conjunto de prueba:", X_test.shape)
    print("\nPrimer registro normalizado:")
    print(X_train[0])

    best_model, best_history, best_config, best_accuracy, resultados_df = search_best_model(
        X_train, y_train, X_test, y_test
    )

    print("\n" + "=" * 70)
    print("MEJOR MODELO")
    print("=" * 70)
    print("Arquitectura:", best_config[0])
    print("Learning Rate:", best_config[1])
    print("Batch Size:", best_config[2])
    print(f"Accuracy: {best_accuracy * 100:.2f}%")

    model = best_model
    history = best_history

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\n==============================")
    print("RESULTADOS EN PRUEBA")
    print("==============================")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    y_prob = model.predict(X_test, verbose=0).ravel()

    best_threshold, best_acc_threshold = find_best_threshold(y_test, y_prob)
    print("\nMejor Threshold:", round(best_threshold, 2))
    print("Accuracy con Threshold:", f"{best_acc_threshold * 100:.2f}%")

    y_pred = (y_prob >= best_threshold).astype(int)
    final_acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy final: {final_acc * 100:.2f}%")

    report_text = classification_report(y_test, y_pred)
    print("\nReporte de clasificación")
    print(report_text)

    save_text_file(
        "accuracy.txt",
        f"Accuracy final: {final_acc * 100:.2f}%\nBest threshold: {best_threshold:.2f}\n",
    )
    save_text_file("classification_report.txt", report_text)

    plot_confusion_matrix(y_test, y_pred)
    roc_auc = plot_roc_curve(y_test, y_prob)
    plot_loss_curve(history)

    best_model.save("modelo.keras")
    print("\nModelo guardado correctamente.")
    print(
        "Archivos generados: confusion_matrix.png, roc_curve.png, loss_curve.png, "
        "accuracy.txt, classification_report.txt, resultados_modelos.csv, modelo.keras"
    )
    print(f"AUC ROC: {roc_auc:.3f}")


if __name__ == "__main__":
    main()
