import matplotlib.pyplot as plt 
import pandas as pd 

optimizer_names = ['SGD', 'Lookahead', 'AdamW']
dirs = ['CIFAR\\' + optimizer_name + '_log.pt' for optimizer_name in optimizer_names]

plt.figure()
for opt_name, direc in zip(optimizer_names, dirs):
    history = pd.read_csv(direc, index_col='epoch')
    plt.plot(history['acc'], label=opt_name)

plt.grid(True)
plt.legend()
plt.show()