"""
File: proto-finetune.py
------------------
Finetunes proto linear probe. 
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
import argparse
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import pdb
import rsbox 
from rsbox import ml, misc
import proto_pretrain 
from proto_pretrain import ProtoEncoder


# ----------------- Global config vars ----------------- #


train_folder = "/mnt/disks/proto/stl_10/train" 
test_folder = "/mnt/disks/proto/stl_10/test"
batch_size = 128  # should be same as baseline 
latent_dim = 128 * 4
output_dim = 128
logger = None
normalize = True
resize = None
experiment_name = "stl10-proto-finetune" + "-" + misc.timestamp()
num_classes = 10
pretrained_encoder_path = "saved/11-10-AM-Jun-08-2023.pth"  # main 
#pretrained_encoder_path = "saved/1-13-PM-Jun-08-2023.pth"
freeze = False
probe = 'mlp'

print("probe type: ", probe)
print("frozen?: ", freeze)
print("batch size: ", batch_size)
print("normalization: ", normalize)
print("latent dim: ", latent_dim)
print("output dim: ", output_dim)



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
    def __init__(self, latent_dim, output_dim, num_classes, pretrained_encoder_path=None, freeze=True):
        super().__init__()
        # encoder 
        self.encoder_module = ProtoEncoder(latent_dim, output_dim)

        # load if pretrained encoder path is provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)

        # remove g projection head 
        self.encoder_module.encoder.fc = nn.Identity()  # Removing projection head g(.)
        self.encoder_module.eval()

        # freeze 
        if freeze:
            self.freeze_encoder()

        # linear head 
        self.linear_head = nn.Linear(latent_dim, num_classes)

    
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


class MLPProbe(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes, pretrained_encoder_path=None, freeze=True):
        super().__init__()
        # encoder 
        self.encoder_module = ProtoEncoder(latent_dim, output_dim)

        # load if pretrained encoder path is provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)

        # remove g projection head 
        self.encoder_module.encoder.fc = nn.Identity()  # Removing projection head g(.)
        self.encoder_module.eval()

        # freeze 
        if freeze:
            self.freeze_encoder()

        # linear head 
        self.linear_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    
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

class ProtoFineTuneExperiment(torchplate.experiment.Experiment):
    def __init__(self): 

        print(f"Running proto FineTuneExperiment. Run name: {experiment_name}")

        assert probe == "mlp" or probe == "linear"

        if probe == "linear":
            self.model = LinearProbe(
                latent_dim, 
                output_dim, 
                num_classes, 
                pretrained_encoder_path=pretrained_encoder_path, 
                freeze=freeze
            )
        elif probe == "mlp":
            self.model = MLPProbe(
                latent_dim, 
                output_dim, 
                num_classes, 
                pretrained_encoder_path=pretrained_encoder_path, 
                freeze=freeze
            )
        else:
            self.model = LinearProbe(
                latent_dim, 
                output_dim, 
                num_classes, 
                pretrained_encoder_path=pretrained_encoder_path, 
                freeze=freeze
            )
        
        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Experiment running on device {self.device}") 
        self.model.to(self.device)

        # training vars 
        if freeze:
            self.optimizer = optim.Adam(self.model.linear_head.parameters(), lr=0.001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # data 
        self.transforms = None
        self.trainset = ImageClassificationDataset(train_folder, transform=self.transforms)
        self.testset = ImageClassificationDataset(test_folder, transform=self.transforms)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False)
        print(f"train set len ~{len(self.trainloader) * batch_size}")
        print(f"test set len {len(self.testloader)}")
        

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True
        )
    


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
        logger = wandb.init(project = "proto231n-new", name = experiment_name)
        
    
    # train 
    exp = ProtoFineTuneExperiment()
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
