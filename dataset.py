from torch.utils.data import Dataset
import random
import torch


class NumberDataset(Dataset):
    """
    Simple dataset that returns two numbers as input, and their addition as output.
    """
    def __init__(self, max_seq_len = 128, max_len=10, min_len=1, size=100000, use_prompt=False):
        super(NumberDataset, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_len = max_len
        self.min_len = min_len
        self.size = size
        self.tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '<pad>', '<eos>', '-', '>', ',']
        self.n_tokens = len(self.tokens)
        self.use_prompt = use_prompt

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        len1, len2 = random.randint(self.min_len, self.max_len), random.randint(self.min_len, self.max_len)

        # simpler for now
        len2 = len1
        num1 = ''.join(random.choice('0123456789') for _ in range(len1))
        num2 = ''.join(random.choice('0123456789') for _ in range(len2))
        return self.encode(int(num1), int(num2))

    def encode(self, num1: int, num2: int):
        """
        Encode the numbers as a sequence of integers.
        """
        string = self.get_string(num1, num2)
        encoding = [self.tokens.index(x) for x in string] + \
                   [self.tokens.index('<eos>')]
        encoding += [self.tokens.index('<pad>')] * (self.max_seq_len + 1 - len(encoding))
        encoding = encoding[:self.max_seq_len + 1]
        return torch.tensor(encoding)

    def get_string(self, num1: int, num2: int):
        if not self.use_prompt:
            return str(num1) + '+' + str(num2) + '=' + str(num1 + num2)

        string = str(num1) + '+' + str(num2) + '->'
        carry_over = 0
        for a, b in zip(reversed(str(num1)), reversed(str(num2))):
            string += a + '+' + b + '+' + str(carry_over) + '=' + str(int(a) + int(b) + carry_over) + ','
            carry_over = (int(a) + int(b) + carry_over) // 10
        string = string[:-1]
        string += '->' + str(int(num1) + int(num2))
        return string
