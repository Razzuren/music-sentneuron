import os
import json
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model

GENERATED_DIR = './generated'

import tensorflow as tf
import numpy as np


def softmax_with_temperature(logits, temperature=1.0):
    """Aplica softmax com temperatura para ajustar a aleatoriedade."""
    logits = np.asarray(logits) / temperature
    exp_logits = np.exp(logits - np.max(logits))  # Evita overflow numérico
    return exp_logits / np.sum(exp_logits)


def sample_with_temperature(predictions, k=5, temperature=1.0):
    """Amostra um próximo token usando temperatura."""
    top_k = tf.math.top_k(predictions, k)
    top_k_indices = top_k.indices.numpy().squeeze()
    top_k_values = top_k.values.numpy().squeeze()

    # Aplica temperatura para suavizar ou acentuar diferenças nas probabilidades
    probs = softmax_with_temperature(top_k_values, temperature)

    # Escolhe um índice com base nas probabilidades ajustadas
    predicted_id = np.random.choice(top_k_indices, p=probs)
    return predicted_id


def process_initial_input(model, init_tokens, char2idx):
    """Executa um passo inicial no modelo para ajustar os estados internos."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            layer.reset_states()

    predictions = None
    for token in init_tokens.split(" "):
        if token in char2idx:
            input_eval = tf.expand_dims([char2idx[token]], 0)
            predictions = model(input_eval)
        else:
            print(f"Aviso: Token desconhecido ignorado -> {token}")

    return predictions


def generate_midi_sequence(model, char2idx, idx2char, init_text="", seq_len=256, k=5, temperature=1.0):
    """Gera uma sequência MIDI que mantém coerência musical."""
    midi_generated = []
    note_count = 0  # Conta quantas notas já foram geradas

    # Configuração inicial do modelo
    predictions = process_initial_input(model, init_text, char2idx)

    for _ in range(seq_len):
        predictions = tf.squeeze(predictions, 0).numpy()
        predicted_id = sample_with_temperature(predictions, k, temperature)

        # Obtém o token correspondente
        token = idx2char.get(predicted_id, "T4")  # Se falhar, gera um tempo padrão

        # Evita sequências que só geram tempos
        if len(midi_generated) > 6 and all(t.startswith("T") for t in midi_generated[-6:]):
            token = f"NOTE_ON_{np.random.randint(50, 80)}_VEL{np.random.randint(60, 110)}"

        # Se houver muitas notas seguidas sem tempo, insere um tempo
        if note_count > 5 and not token.startswith("T"):
            token = f"T{np.random.randint(1, 8)}"
            note_count = 0  # Reseta a contagem

        # Atualiza contadores
        if token.startswith("NOTE_ON"):
            note_count += 1

        midi_generated.append(token)

        # Passa o novo token como entrada ao modelo
        input_eval = tf.expand_dims([predicted_id], 0)
        predictions = model(input_eval)

    return init_text + " " + " ".join(midi_generated)

if __name__ == "__main__":
    """Execução principal"""
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint do modelo.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON com char2idx.")
    parser.add_argument('--embed', type=int, required=True, help="Tamanho do embedding.")
    parser.add_argument('--units', type=int, required=True, help="Quantidade de unidades LSTM.")
    parser.add_argument('--layers', type=int, required=True, help="Número de camadas LSTM.")
    parser.add_argument('--seqinit', type=str, default="NOTE_ON_60_VEL100", help="Sequência inicial.")
    parser.add_argument('--seqlen', type=int, default=256, help="Comprimento da sequência.")
    parser.add_argument('--topk', type=int, default=3, help="Número de candidatos para amostragem.")
    parser.add_argument('--output', type=str, required=True, help="Arquivo de saída.")
    opt = parser.parse_args()

    # Carrega char2idx de um JSON
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Cria idx2char reverso
    idx2char = {idx: char for char, idx in char2idx.items()}

    # Obtém tamanho do vocabulário
    vocab_size = len(char2idx)

    # Reconstrói o modelo e carrega os pesos
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(opt.model)
    model.build(tf.TensorShape([1, 0]))

    # Gera MIDI como texto
    midi_txt =midi_generated = generate_midi_sequence(model, char2idx, idx2char,
                                        init_text="NOTE_ON_60_VEL100",
                                        seq_len= opt.embed,
                                        k=5,
                                        temperature=0.8)
    print(midi_txt)

    # Escreve para um arquivo MIDI usando o encoder
    me.write(midi_txt, opt.output)