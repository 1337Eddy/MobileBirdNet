from cgi import test
import numpy as np
import torch

from pathlib import Path
from torch.autograd import Variable
import torch.optim as optim
from monitor import Monitor
from monitor import Status

from utils.metrics import AverageMeter
from utils.metrics import accuracy

class AnalyzeMobileBirdnet():
    def __init__(self, model, test_loader, train_loader, save_path, dataset, criterion, lr):
        torch.cuda.manual_seed(1337)
        torch.manual_seed(73)
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader 
        self.save_path = save_path
        self.dataset = dataset
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def prepare_data_and_labels(self, data, target):
        data = data.cuda(non_blocking=True)    
        data = Variable(data)       
        target = self.dataset.labels_to_one_hot_encondings(target)
        target= torch.from_numpy(target)
        target = target.cuda(non_blocking=True)
        target = Variable(target)
        return data, target

    def train(self):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        for idx, (data, target) in enumerate(self.train_loader):
            torch.cuda.empty_cache()

            data, target = self.prepare_data_and_labels(data, target)

            output = self.model(data.float())
            output = np.squeeze(output)
            self.optimizer.zero_grad()
            loss = self.criterion(output.float(), target.float())
            loss.backward()
            self.optimizer.step()
            #Calculate and update metrics
            losses.update(loss.item(), data.size(0))
            prec = accuracy(output.data, target)
            top1.update(prec, data.size(0))

        return losses, top1

    def test(self):        
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        for data, target in self.test_loader:
            torch.cuda.empty_cache()

            data, target = self.prepare_data_and_labels(data, target)

            #Run model
            output = self.model(data.float())
            output = np.squeeze(output)
            
            loss = self.criterion(output.float(), target.float())

            #Calculate and update metrics
            losses.update(loss.item(), data.size(0))
            prec = accuracy(output.data, target)
            top1.update(prec, data.size(0))
        return losses, top1


    def save_model(self, epochs, birdnet, optimizer, val_loss, val_top1, 
                train_loss_list, test_loss_list, train_acc_list, test_acc_list, path):
        Path(path[:-len(path.split('/')[-1])]).mkdir(parents=True, exist_ok=True)
        torch.save({
                'train_loss_list': train_loss_list,
                'test_loss_list': test_loss_list,
                'train_acc_list': train_acc_list,
                'test_acc_list': test_acc_list,
                'epoch': epochs,
                'model_state_dict': birdnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_top1
                }, path)

    def start_training(self, epochs):
        version = 0
        print("Start training process")
        monitoring = Monitor(1, 2)
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []

        for i in range(0, epochs):
            train_loss, train_top1 = self.train()
            test_loss, test_top1 = self.test()

            train_loss_list.append(train_loss.avg)
            test_loss_list.append(test_loss.avg)
            train_acc_list.append(train_top1.avg)
            test_acc_list.append(test_top1.avg) 

            status = monitoring.update(test_loss.avg, lr=self.lr)

            print('epoch: {:d} \ntrain loss avg: {train_loss.avg:.4f}, accuracy avg: {train_top1.avg:.4f}\t'
                  '\ntest loss avg: {test_loss.avg:.4f}, accuracy avg: {test_top1.avg:.4f}'.format(i, train_loss=train_loss,train_top1=train_top1, test_loss=test_loss, test_top1=test_top1))

            if (status == Status.LEARNING_RATE):
                self.lr *= 0.5
            elif (status == Status.STOP):
                break 

            if (i % 5 == 0):
                print("Save checkpoint: " + self.save_path + "birdnet_v" + str(version) + ".pt")
                self.save_model(i, self.model, self.optimizer, test_loss, test_top1, 
                    train_loss_list, test_loss_list, train_acc_list, test_acc_list, 
                    self.save_path + "birdnet_v" + str(version) + ".pt")       
                version += 1
        self.save_model(i, self.model, self.optimizer, test_loss, test_top1, train_loss_list, test_loss_list, 
        train_acc_list, test_acc_list, self.save_path  + "birdnet_final.pt")       
        print("Saved Model!")