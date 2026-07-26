import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

TRAIN_SIZE = 20000
VALIDATION_SPLIT = 0.1
MAX_EPOCHS = 8
BATCH_SIZE = 128
PATIENCE = 3


def load_data():
    # MNIST já vem dividido em treino e teste; aqui só normalizamos e ajustamos o canal.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = keras.Sequential(name="mnist_cnn")
    model.add(keras.Input(shape=(28, 28, 1)))

    for filters in (16, 32, 64):
        model.add(layers.Conv2D(filters, 3, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(filters, 3, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D())

    model.add(layers.Dropout(0.4))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")

    keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train[:TRAIN_SIZE]
    y_train = y_train[:TRAIN_SIZE]
    print(f"Treinando com {TRAIN_SIZE} amostras do MNIST")

    model = build_model()

    callbacks = [
        # EarlyStopping interrompe o treino quando a validação deixa de melhorar.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    final_val_accuracy = history.history["val_accuracy"][-1]
    print(f"Acurácia final de validação: {final_val_accuracy:.4f}")

    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
