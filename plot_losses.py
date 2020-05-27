import matplotlib.pyplot as plt 
import pandas as pd 

optimizer_names = ['SGD', 'Lookahead', 'AdamW', 'Polyak']
dirs = ['CIFAR\\' + optimizer_name + '_log.pt' for optimizer_name in optimizer_names]

fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(12,12))
for opt_name, direc in zip(optimizer_names, dirs):
    history = pd.read_csv(direc, index_col='epoch')
    ax[0,0].plot(history['loss'], label=opt_name)
    ax[0,1].plot(history['acc'], label=opt_name)
    ax[1,0].plot(history['val_loss'], label=opt_name)
    ax[1,1].plot(history['val_acc'], label=opt_name)

ax[0,0].set_title("Train Loss")
ax[0,1].set_title("Train Acc")
ax[1,0].set_title("Val Loss")
ax[1,1].set_title("Val Acc")

ax[0,0].grid(True)
ax[0,1].grid(True)
ax[1,0].grid(True)
ax[1,1].grid(True)

ax[1,0].set_ylim(0,1.5)

ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()

plt.show()