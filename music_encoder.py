import os
import argparse
import mido


def midi_to_tokens(midi_file):
    """Converte um arquivo MIDI para uma string de tokens"""
    mid = mido.MidiFile(midi_file)
    tokens = []
    current_time = 0

    for track in mid.tracks:
        for msg in track:
            if msg.time > 0:
                tokens.append(f"T{msg.time}")  # Tempo relativo entre eventos

            if msg.type == 'note_on' and msg.velocity > 0:
                tokens.append(f"NOTE_ON_{msg.note}_VEL{msg.velocity}")
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                tokens.append(f"NOTE_OFF_{msg.note}")

    return " ".join(tokens)  # Retorna string única de tokens


def tokens_to_midi(token_str, output_midi, ticks_per_beat=384):
    """Reconstrói um arquivo MIDI a partir de uma string de tokens, preservando PPQ"""
    tokens = token_str.split()
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)  # Define os ticks por batida
    track = mido.MidiTrack()
    mid.tracks.append(track)

    current_time = 0

    for token in tokens:
        if token.startswith("NOTE_ON_"):
            parts = token.split("_")
            note = int(parts[2])
            velocity = int(parts[3].replace("VEL", ""))
            track.append(mido.Message("note_on", note=note, velocity=velocity, time=current_time))
            current_time = 0  # Reset do tempo para acordes funcionarem

        elif token.startswith("NOTE_OFF_"):
            parts = token.split("_")
            note = int(parts[2])
            track.append(mido.Message("note_off", note=note, velocity=0, time=current_time))
            current_time = 0  # Reset do tempo para acordes funcionarem

        elif token.startswith("T"):
            current_time = int(token[1:])  # Atualiza o tempo relativo

    mid.save(output_midi)



def load(datapath):
    """Processa arquivos individuais ou diretórios, retornando texto e vocabulário"""
    text = ""
    vocab = set()

    if os.path.isfile(datapath):
        # Processa um único arquivo
        text = process_file(datapath)
        vocab = set(text.split(" "))
    else:
        # Processa todos os arquivos dentro da pasta
        for file in os.listdir(datapath):
            file_path = os.path.join(datapath, file)
            if os.path.isfile(file_path) and file_path.endswith((".mid", ".txt")):
                encoded_data = process_file(file_path)
                vocab.update(encoded_data.split(" "))
                text += encoded_data + " "

        text = text.strip()

    return text, vocab


def process_file(file_path):
    """Lida com arquivos individuais e converte MIDI para tokens ou vice-versa"""
    file_extension = os.path.splitext(file_path)[1]

    if file_extension == ".mid":
        return midi_to_tokens(file_path)
    elif file_extension == ".txt":
        with open(file_path, "r") as f:
            return f.read().strip()
    else:
        print(f"Ignorado: {file_path} (Formato não suportado)")
        return ""


def write(encoded_midi, output_path):
    """Salva os tokens em um arquivo ou converte para MIDI"""
    if output_path.endswith(".mid"):
        tokens_to_midi(encoded_midi, output_path)
    else:
        with open(output_path, "w") as f:
            f.write(encoded_midi)


def main():
    """Função principal que lida com argumentos"""
    parser = argparse.ArgumentParser(description='midi_encoder.py')
    parser.add_argument('--input', type=str, required=True, help="Arquivo ou pasta de entrada.")
    parser.add_argument('--output', type=str, required=True, help="Arquivo ou pasta de saída.")
    opt = parser.parse_args()

    text, _ = load(opt.input)

    if os.path.isdir(opt.input):
        # Para pastas, salva cada arquivo processado separadamente
        os.makedirs(opt.output, exist_ok=True)
        for file in os.listdir(opt.input):
            file_path = os.path.join(opt.input, file)
            output_file = os.path.join(opt.output,
                                       file.replace(".mid", ".txt") if file.endswith(".mid") else file.replace(".txt",
                                                                                                               ".mid"))
            encoded_text = process_file(file_path)
            write(encoded_text, output_file)
            print(f"Convertido: {file_path} → {output_file}")

    else:
        # Para arquivos únicos, salva diretamente
        write(text, opt.output)
        print(f"Convertido: {opt.input} → {opt.output}")


if __name__ == "__main__":
    main()
