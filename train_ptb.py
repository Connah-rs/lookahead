from PTB.model import RNNModel
from PTB.utils import repackage_hidden
import torch 
from torch.utils.data import DataLoader
from torchtext.datasets import PennTreebank
import torchbearer
# import chainer

device = 'cuda:0'

BPTT_LEN = 70
BATCH_SIZE = 80
NB_EPOCHS = 3

train_data, val_data, test_data = PennTreebank.iters(batch_size=BATCH_SIZE, bptt_len=BPTT_LEN)#, device=device)


model = RNNModel('LSTM', 5000, 400, 1150, 3, 0.4, 0.3, 0.65, 0.1, 0.5, True)#.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)

hidden = model.init_hidden(BATCH_SIZE)
# model.train()
for i, item in enumerate(train_data):
    inputs, labels = item.text, item.target
    hidden = repackage_hidden(hidden)
    optimizer.zero_grad()
    outputs = model(inputs, hidden, return_h=True)
    loss = criterion(outputs, labels)
    loss.backward()
    optimiser.step()


# trial = torchbearer.Trial(model, optimizer, criterion, metrics=['loss', 'accuracy']).to(device)
# trial.with_generators(trainloader, val_generator=valloader)
# results.append(trial.run(epochs=10))


# my
# print("Model built")


# train, val, test = chainer.datasets.get_ptb_words()
# n_vocab = max(train) + 1
# print(n_vocab)

# print(train.shape, val.shape, test.shape)