from matplotlib import rcParams
import matplotlib.pyplot as plt
import re

rcParams['font.family']='sans-serif'
rcParams['font.sans-serif']='SimSun,Times New Roman'

logFile=r'log_tvt.txt'
text=''
file=open(logFile)
train_loss=[]
train_acc=[]
vaid_acc=[]
test_acc=[]

def acc_show(train_acc,test_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    plt.xlabel('times')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

def loss_show(train_loss):
    plt.plot(train_loss, label='train_loss')
    # plt.plot(test_acc, label='test_acc')
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()

def plit_show():
    for line in file:
        if(line!='Finished Training'):
            cut_line=line.split('train_loss:')[1].split('test_accuracy:')[0]
            train_loss.append(float(cut_line[:6]))
        # print(cut_line[:6])
            cut_line = line.split('test_accuracy:')[1].split(' train_accuracy:')[0]
            test_acc.append(float(cut_line[:6]))
            cut_line = line.split('train_accuracy:')[1].split(' vaid_accuracy')[0]
            train_acc.append(float(cut_line[:6]))
            cut_line = line.split('vaid_accuracy:')[1]
            train_acc.append(float(cut_line))
    acc_show(train_acc,test_acc)
    # loss_show(train_loss)


if __name__=="__main__":
    plit_show()