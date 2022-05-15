from torch.utils.data import Dataset
import random
import torch


class NumberDataset(Dataset):
    """
    Simple dataset that returns two numbers as input, and their addition as output.
    """
    def __init__(self, max_seq_len = 128, max_len=10, min_len=1, size=100000):
        super(NumberDataset, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_len = max_len
        self.min_len = min_len
        self.size = size
        self.tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '<pad>', '<eos>']
        self.n_tokens = len(self.tokens)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        len1, len2 = random.randint(self.min_len, self.max_len), random.randint(self.min_len, self.max_len)
        num1 = ''.join(random.choice('0123456789') for _ in range(len1))
        num2 = ''.join(random.choice('0123456789') for _ in range(len2))
        return self.encode(int(num1), int(num2))

    def encode(self, num1: int, num2: int):
        """
        Encode the numbers as a sequence of integers.
        """
        encoding = [self.tokens.index(x) for x in str(num1) + '+' + str(num2) + '=' + str(num1 + num2)] + \
                   [self.tokens.index('<eos>')]
        encoding += [self.tokens.index('<pad>')] * (self.max_seq_len + 1 - len(encoding))
        encoding = encoding[:self.max_seq_len + 1]
        return torch.tensor(encoding)
