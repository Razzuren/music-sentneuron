import tensorflow as tf
import numpy as np
import json
import argparse

from train_generative import build_generative_model


def test_single_prediction(model, char2idx, idx2char, init_text="NOTE_ON_60_VEL100"):
    """Executa um único predict() e imprime a saída"""

    # Converte o texto inicial para IDs
    input_eval = tf.expand_dims([char2idx[token] for token in init_text.split(" ") if token in char2idx], 0)

    # Reseta os estados da LSTM para evitar influência de previsões anteriores
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            layer.reset_states()

    # Executa um único passo de predição
    predictions = model.predict(input_eval, verbose=0)

    # Mostra a saída bruta da predição
    print("\n🔎 Previsão Bruta (logits):", predictions)

    # Obtém o índice com maior probabilidade (argmax)
    predicted_id = np.argmax(predictions, axis=-1)[0, -1]

    # Converte o índice para token MIDI
    predicted_token = idx2char.get(predicted_id, "T4")  # Se não encontrar, usa tempo padrão

    print(f"\n🎵 Token previsto: {predicted_token}")

    return predicted_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testa um único predict() no modelo MIDI')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint do modelo.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON com char2idx.")
    parser.add_argument('--embed', type=int, required=True, help="Tamanho do embedding.")
    parser.add_argument('--units', type=int, required=True, help="Quantidade de unidades LSTM.")
    parser.add_argument('--layers', type=int, required=True, help="Número de camadas LSTM.")
    parser.add_argument('--seqinit', type=str, default="NOTE_ON_60_VEL100", help="Sequência inicial.")
    opt = parser.parse_args()

    # Carrega char2idx de um JSON
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Cria idx2char reverso
    idx2char = {idx: char for char, idx in char2idx.items()}

    # Obtém tamanho do vocabulário
    vocab_size = len(char2idx)

    # Reconstrói o modelo e carrega os pesos
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=64)
    model.load_weights(opt.model)
    model.build(tf.TensorShape([64, 0]))

    # Testa um único predict()
    test_single_prediction(model, char2idx, idx2char, opt.seqinit)
