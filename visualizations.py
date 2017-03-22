"""
This file containes helper function, mainly for visualization.

Author: ksopyla (https://github.com/ksopyla)
"""

import matplotlib.pyplot as plt



def losses_accuracies_plots(train_losses, train_acc, test_losses, test_acc,plot_title="Loss, train acc, test acc",step=100):
    """
    Function generates matplolib plots with loss and accuracies
    
    
    Parameters
    ----------
    losses : list
        list with values of loss function, computed every 'step' trainning iterations
    train_acc : list
        list with values  of training acuracies, computed every 'step' trainning iterations
    test_acc : list
        list with values  of testing acuracies, computed every 'step' trainning iterations
    step : int
        number of trainning iteration after which we compute (loss and accuracies)
    plot_title: string
        title of the plot
    
    Raises
    ------
    Exception
        when an error occure
    """
        
    training_iters = len(train_losses)
    # iters_steps
    iter_steps = [step *k for k in range(training_iters)]

    imh = plt.figure(1, figsize=(15, 14), dpi=160)
    # imh.tight_layout()
    # imh.subplots_adjust(top=0.88)

    imh.suptitle(plot_title)
    plt.subplot(221)
    #plt.plot(iter_steps,losses, '-g', label='Loss')
    plt.semilogy(iter_steps, train_losses, '-g', label='Trn Loss')
    plt.title('Train Loss ')
    plt.subplot(222)
    plt.plot(iter_steps, train_acc, '-r', label='Trn Acc')
    plt.title('Train Accuracy')

    plt.subplot(223)
    plt.semilogy(iter_steps, test_losses, '-g', label='Tst Loss')
    plt.title('Test Loss')
    plt.subplot(224)
    plt.plot(iter_steps, test_acc, '-r', label='Tst Acc')
    plt.title('Test Accuracy')


    #plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.savefig("./plots/"+plot_title)
    plt.show()