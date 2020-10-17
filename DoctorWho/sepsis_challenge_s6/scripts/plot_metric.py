import json 
import numpy as np 
import argparse
from IPython import embed
import sys
import os 

def plot_metric(title, path, show=False):

    if show:
        import matplotlib.pyplot as plt 
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt 

    
    with open(path, 'rb') as f:
        data = json.load(f)
    #embed() 

    train_loss = data['train_loss']['values']
    validation_loss = data['validation_loss']['values']
    validation_utility = data['validation_UTILIY']['values']
    overall_max = max(max(train_loss), max(validation_loss))
    max_epoch = len(validation_loss)/float(4)
    x = np.arange(0, max_epoch, 0.25) # steps in epochs, validating 4x per epoch..
    x += .25

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    
    head, tail = os.path.split(title) #extract title name from title path
    title_name = tail 
    ax1.set_title(title_name)

    col1 = 'red'
    col2 = 'blue'
    col3 = 'green' 

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=col1)
    l1, = ax1.plot(x, train_loss, color=col1)
    ax1.tick_params(axis='y', labelcolor=col1)

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    #ax2.set_ylabel('Validation Loss', color=col2) 
    #l2, = ax2.plot(x, validation_loss, color=col2)

    ax1.set_ylim((0,overall_max))
    #ax2.set_ylim((0,overall_max))

    ax3 = ax1.twinx()
    ax3.set_ylabel('Validation Utility', color=col3)
    l3, = ax3.plot(x, validation_utility, color=col3)

    plt.legend((l1, l3), ('Train Loss', 'Validation Utility')) 

    plt.savefig(title +'.pdf')
    if show:
        plt.show()



if __name__ == "__main__":
    
    #Parse Arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True, 
                        help="Full path to the metrics to plot")
    parser.add_argument("--title", required=True, 
                        help="Plotting title with path")
    parser.add_argument('--show', help='show plots (use only with graphics device)',
        action='store_true')
    args = parser.parse_args()


    path = args.data_path
    title = args.title
    plot_metric(title, path)

'''
# on same axis:
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

'''
