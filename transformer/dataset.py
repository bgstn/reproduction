import numpy as np


class Dataset:
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, num_samples):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.num_samples = num_samples

    def generate(self):
        src_sent = np.random.randint(low=1, high=self.src_vocab_size, size=(self.num_samples, self.src_seq_len))
        tgt_sent = np.roll(src_sent, shift=1, axis=-1)
        tgt_input = tgt_sent[:, :(self.tgt_seq_len-1)]
        tgt_output = tgt_sent[:, 1:]
        return src_sent, tgt_input, tgt_output
