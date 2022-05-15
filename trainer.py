from model import Model
from dataset import NumberDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import tqdm

dataset = NumberDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
eval_dataset = NumberDataset(max_len=11, min_len=11, size=10000)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)
model = Model(max_seq_len=dataset.max_seq_len, n_tokens=dataset.n_tokens, dim=128, depth=12, heads=8, dim_head=16,
              mlp_dim=512).to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):

    tqdm_dataloader = tqdm.tqdm(dataloader)

    train_acc = 0
    train_loss = 0
    train_samples = 0

    for x in tqdm_dataloader:
        x = x.to('cuda')
        pred = model(x[:, :-1])
        n_tokens = pred.shape[-1]
        loss = F.cross_entropy(pred.view(-1, n_tokens), x[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc = (pred.argmax(dim=2) == x[:, 1:]).float().mean()
        train_acc += acc.item() * len(x)
        train_loss += loss.item() * len(x)
        train_samples += len(x)
        tqdm_dataloader.set_description(f"Epoch {epoch} | Loss: {train_loss / train_samples:.4f} | Acc: "
                                        f"{train_acc / train_samples:.4f}")

    tqdm_dataloader.close()

    eval_acc = 0
    eval_loss = 0
    eval_samples = 0

    tqdm_eval_dataloader = tqdm.tqdm(eval_dataloader)

    for x in tqdm_eval_dataloader:
        x = x.to('cuda')
        pred = model(x[:, :-1])
        n_tokens = pred.shape[-1]
        loss = F.cross_entropy(pred.view(-1, n_tokens), x[:, 1:].contiguous().view(-1))

        acc = (pred.argmax(dim=2) == x[:, 1:]).float().mean()
        eval_acc += acc.item() * len(x)
        eval_loss += loss.item() * len(x)
        eval_samples += len(x)
        tqdm_eval_dataloader.set_description(f"Epoch {epoch} | Loss: {eval_loss / train_samples:.4f} | Acc: "
                                        f"{eval_acc / train_samples:.4f}")

    tqdm_eval_dataloader.close()
