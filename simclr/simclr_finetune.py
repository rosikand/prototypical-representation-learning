"""
File: simclr-finetune.py
------------------
Finetunes simclr linear probe. 
"""


import torchplate
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
import sys
import os
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import requests
import argparse
import cloudpickle as cp
from urllib.request import urlopen
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import pdb
import rsbox 
from rsbox import ml, misc



# ----------------- Global config vars ----------------- #

data_folder_path = root_path = "/mnt/disks/proto/stl10_images_rotated/"
batch_size = 64
logger = None
experiment_name = "simclr-finetune" + "-" + misc.timestamp()
latent_dim = 128
num_classes = 10
pretrained_encoder_path = "saved/1-56-PM-Jun-04-2023.pth"
apply_transforms = True
latent_dim_multiplier = 4
normalize = False
resize = None

# ----------------- Dataset ----------------- #


class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        if ".DS_Store" in self.classes:
            self.classes.remove(".DS_Store")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                if not os.path.isfile(image_path):
                    continue

                images.append((image_path, self.class_to_idx[class_name]))

        return images

        
    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = ml.load_image(image_path, resize=resize, normalize=normalize) 

        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float)
        
        if not torch.is_tensor(label):
            label = torch.tensor(label)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.dtype != torch.float:
            image = image.to(torch.float)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)



# ----------------- Models ----------------- #


class LinearProbe(nn.Module):
    """
    Helper class which functionizes two separate PyTorch modules
    (encoder + linear probe). Inherits from torch.nn.Module so 
    instantiations of this class are trainable PyTorch modules. 
    Functionally, pre-train the encoder_module and then fine-tune
    this linear probe. Make sure to freeze the encoder_module 
    before gradient descent fine-tuning. 
    Parameters:
        - encoder_module: a proto-pretrain PyTorch module which takes in an image
            and outputs a latent vector.
        - pretrained_encoder_path: path to a proto pretrained encoder module.
        - latent_dim: the input dimension of the linear probe. i.e. the encoder output dimension
        - num_classes: the number of classes in the dataset to be classified. 
    """

    def __init__(self, encoder_module, latent_dim, num_classes, pretrained_encoder_path=None, freeze=True):
        super().__init__()
        # encoder 
        self.encoder_module = encoder_module

        # load if pretrained encoder path is provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)

        # remove g projection head 
        self.encoder_module.fc = nn.Identity()  # Removing projection head g(.)
        self.encoder_module.eval()

        # freeze 
        if freeze:
            self.freeze_encoder()

        # linear head 
        self.linear_head = nn.Linear(latent_dim * latent_dim_multiplier, num_classes)

    
    def encode(self, x):
        return self.encoder_module(x)
    
    def head(self, x):
        return self.linear_head(x)


    def freeze_encoder(self):
        # Freeze the encoder_module params 
        for param in self.encoder_module.parameters():
            param.requires_grad = False
        
        print("Encoder model params frozen!")
    

    def load_pretrained_encoder(self, encoder_path):
        '''Load a pretrained encoder from a file.'''
        self.encoder_module.load_state_dict(torch.load(encoder_path))
        print(f"successfully loaded the pretrained encoder from {encoder_path}")

    
    def forward(self, x):
        latent = self.encode(x)
        return self.head(latent)



# ----------------- Training Experiments ----------------- #

class FineTuneExperiment(torchplate.experiment.Experiment):
    def __init__(self): 

        print(f"Running SimCLR FineTuneExperiment. Run name: {experiment_name}")

        self.hidden_dim = latent_dim
        enc = self.get_encoder_obj(self.hidden_dim)
    
        self.model = LinearProbe(
            enc, 
            self.hidden_dim, 
            num_classes, 
            pretrained_encoder_path=pretrained_encoder_path, 
            freeze=True
        )
        
        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Experiment running on device {self.device}") 
        self.model.to(self.device)

        # training vars 
        self.optimizer = optim.Adam(self.model.linear_head.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # data 
        if apply_transforms:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transforms = None

        self.dataset = ImageClassificationDataset(data_folder_path, transform=self.transforms)
        torch_set = torchplate.utils.get_xy_dataset(self.dataset)
        torch_sets = torchplate.utils.split_dataset(torch_set)
        self.trainset = torch_sets[0]
        self.testset = torch_sets[1]
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False)


        # inherit from torchplate.experiment.Experiment and pass in
        # model, optimizer, and dataloader 
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True
        )


    def get_encoder_obj(self, hidden_dim):
        model = torchvision.models.resnet18(num_classes = latent_dim_multiplier * hidden_dim) 
        # f(x) -> f(x) + g(x)
        model.fc = nn.Sequential(
            model.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim_multiplier * hidden_dim, hidden_dim)
        )

        return model
    
    
    def plot_samples(self):
        # plot the first few samples from the dataset to visualize 
        for i, (x, y) in enumerate(self.trainloader):
            ml.plot(x[0])  # plot first in batch  
            print(y[0])
            if i == 5:
                break   


    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        loss_val = self.criterion(logits, y)
        acc = tp_metrics.calculate_accuracy(logits, y)
        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        return metrics_dict
    

    def test(self):
        self.model.eval()
        with torch.no_grad():
            acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.testloader)
            for batch in tqdm_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                acc.update(logits, y)
                tqdm_loader.set_description(f'Accuracy: {acc.get()}')
            
            final_acc = acc.get()
            print(f'Test accuracy: {final_acc}')
            acc.reset()

        return final_acc


    def on_epoch_end(self):
        test_acc = self.test()
        if self.wandb_logger is not None:
            self.wandb_logger.log({"Test accuracy": test_acc})

    
    def on_run_end(self):
        self.save_weights()

    def on_epoch_start(self):
        self.model.train()



# ----------------- Runner ----------------- #


def main(args):
    # update globals from cli args 
    if args.name is not None:
        global experiment_name 
        experiment_name = args.name + "-" + experiment_name

    if args.log:
        global logger 
        logger = wandb.init(project = "proto231n", name = experiment_name)
        
    
    # train 
    exp = FineTuneExperiment()
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)
    exp.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?')
    args = parser.parse_args()
    main(args)