import numpy as np


class Dataset:
    def __init__(self):
        pass

    def run(self):
        data_path = 'data/cmn.txt'

        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        for line in lines[: min(100000, len(lines) - 1)]:
            input_text, target_text = line.split('\t')[:2]

            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = 'S' + target_text + 'E'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        self.src_vocab = sorted(list(input_characters))
        self.tgt_vocab = sorted(list(target_characters))
        self.src_vocab_size = len(input_characters)
        self.tgt_vocab_size = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])
        print("======================= src info =======================")
        print("src vocab size: {}".format(self.src_vocab_size))
        print("src max seq length: {}".format(self.max_encoder_seq_length))
        print("\n\n")

        print("======================= tgt info =======================")
        print("tgt vocab size: {}".format(self.tgt_vocab_size))
        print("tgt max seq length: {}".format(self.max_decoder_seq_length))

        self.src_word2idx = dict(
            [(char, i + 1) for i, char in enumerate(input_characters)])
        self.src_idx2word = dict(
            [(i + 1, char) for i, char in enumerate(input_characters)])

        self.tgt_word2idx = dict(
            [(char, i + 1) for i, char in enumerate(target_characters)])
        self.tgt_idx2word = dict(
            [(i + 1, char) for i, char in enumerate(target_characters)])

        src_idxes = []
        tgt_idxes = []
        tgt_onehot = []
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            src_idx = [self.src_word2idx[c] for c in input_text] + (self.max_encoder_seq_length - len(input_text)) * [0]
            tgt_idx = [self.tgt_word2idx[c] for c in target_text if c != 'E'] + \
                      (self.max_decoder_seq_length - len(target_text)) * [0]

            tgt_idx_onehot = []
            for idx, c in enumerate(target_text):
                if c == 'S' and idx == 0:
                    continue
                else:
                    tmp = np.zeros((self.tgt_vocab_size,), dtype='int16')
                    tmp_id = self.tgt_word2idx[c]
                    tmp[tmp_id-1] = 1
                    tmp = tmp.reshape(1, 1, -1)
                    tgt_idx_onehot.append(tmp)

            tgt_idx_onehot += [np.zeros((1, 1, self.tgt_vocab_size), dtype='int8')] * (self.max_decoder_seq_length - len(target_text))
            tgt_idx_onehot = np.concatenate(tgt_idx_onehot, axis=1)
            src_idxes.append(np.array(src_idx, dtype='int8').reshape(1, -1))
            tgt_idxes.append(np.array(tgt_idx, dtype='int8').reshape(1, -1))
            tgt_onehot.append(tgt_idx_onehot)

        self.src_idxes = src_idxes
        self.tgt_idxes = tgt_idxes
        self.tgt_idexes_onehot = tgt_onehot


if __name__ == "__main__":
    dataset = Dataset()
    dataset.run()
    print(dataset.src_idxes[0])
