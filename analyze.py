
import argparse
from cgi import test
import os
import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from data import CallsDataset
from model import MobileBirdNet
from torch import nn

from train_mobilenet import AnalyzeMobileBirdnet

class DataLabels():
    def __init__(self, path):
        self.path = path
        self.birds = os.listdir(path)
        self.birds = sorted(self.birds)
        self.num_classes = len(self.birds)

        self.bird_dict = {x: self.birds.index(x) for x in self.birds}

    def labels_to_one_hot_encondings(sefl, labels):
        result = np.zeros((len(labels), len(sefl.birds)))
        for i in range(0, len(labels)):
            result[i][sefl.bird_dict[labels[i]]] = 1
        return result

    def id_to_label(self, id):
        return list(self.bird_dict)[id]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Set programm into train or evaluation mode')
    parser.add_argument('--load_model', default='', help='Load model from file')
    parser.add_argument('--epochs', default=20, help='Specify number of epochs for training')
    parser.add_argument('--save_path', default='/media/eddy/datasets/models/mobile_net2/', help='Specifies the path where final model and checkpoints are saved')
    parser.add_argument('--lr', default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', default=16, help='Number of samples for one train batch')
    parser.add_argument('--threads', default=16)
    parser.add_argument('--eval_file', default='/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/1calls/arcter/XC582288-326656.wav')
    parser.add_argument('--train_set', default="/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/calls/")
    
    args = parser.parse_args()
    mode = args.mode
    num_workers=int(args.threads)
    batch_size=args.batch_size
    lr=float(args.lr)
    dataset_path = args.train_set

    
    mobile_net = MobileBirdNet()

    mobile_net = nn.DataParallel(mobile_net).cuda() 
    mobile_net = mobile_net.float()

    if (args.load_model != ''):
        checkpoint = torch.load(args.load_model)
        criterion = nn.CrossEntropyLoss().cuda()
        mobile_net.load_state_dict(checkpoint['model_state_dict'])
        



    if (mode == 'train'):
        data = DataLabels(dataset_path + "train/")

        train_dataset = CallsDataset(dataset_path + "train/")
        test_dataset = CallsDataset(dataset_path + "test/")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        criterion = nn.CrossEntropyLoss().cuda()
        #Start Training
        analyze = AnalyzeMobileBirdnet(model=mobile_net, dataset=data, lr=lr, criterion=criterion, train_loader=train_loader, 
                                    test_loader=test_loader, save_path=args.save_path)
        #summary(mobile_net, (1, 64, 384))
        analyze.start_training(int(args.epochs))

if __name__ == '__main__':
    main()