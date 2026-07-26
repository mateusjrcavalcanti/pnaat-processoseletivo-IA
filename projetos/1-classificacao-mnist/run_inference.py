import os

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.h5) usando tf.lite.Interpreter
#   2. Rodar inferência em pelo menos 5 amostras do conjunto de teste do MNIST
#   3. Imprimir no terminal, para cada amostra: classe predita vs. classe real
# ---------------------------------------------------------------------------

N_SAMPLES = 5

def prepare_input(sample, input_details):
    sample = np.expand_dims(sample, axis=0)
    dtype = input_details[0]["dtype"]

    if np.issubdtype(dtype, np.floating):
        return sample.astype(dtype)

    scale, zero_point = input_details[0]["quantization"]
    if scale == 0:
        return sample.astype(dtype)

    quantized = np.round(sample / scale + zero_point)
    return quantized.astype(dtype)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Rodando inferência em {N_SAMPLES} amostras usando {os.path.basename(model_path)}:\n")
    for index in range(N_SAMPLES):
        # Cada amostra é preparada, enviada ao interpretador e lida individualmente.
        sample = prepare_input(x_test[index], input_details)
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_class = int(np.argmax(prediction))

        print(f"Amostra {index + 1}: predito={predicted_class} | real={int(y_test[index])}")


if __name__ == "__main__":
    main()
