"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        losses = []
        val_losses =[]

        for epoch,acc,loss,val_acc,val_loss in reader:
            losses.append(float(loss))
            val_losses.append(float(val_loss))

        plt.title('Loss in Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(losses, label='training loss')
        plt.plot(val_losses, label='validation loss')
        plt.axvline(x=9,color='gray',dashes=(5, 2, 1, 2))
        plt.legend(loc='upper right')
        plt.show()

if __name__ == '__main__':
 
    #training_log = './logs/logs/A-20190525-091727/A-20190525-091727.log'

    #training_log = './logs/logs/B-20190525-091924/B-20190525-091924.log'
    
    training_log = './logs/logs/C-20190525-092130/C-20190525-092130.log'

    #training_log = './logs/logs/D-20190525-092221/D-20190525-092221.log'

    main(training_log)
