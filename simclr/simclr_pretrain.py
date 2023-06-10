"""
File: simclr_pretrain.py
------------------
Implements simclr pre-training. 
Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#Logistic-Regression. 
"""


# implements simclr 
import torchvision
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import os
import torchplate
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
from torch.utils.data import Dataset
from rsbox import ml
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from rsbox import ml, misc
import torch.optim as optim
import argparse



# ----------------- Global config vars ----------------- #

n_views = 2
temperature = 0.07
batch_size = 256
hidden_dim = 128
data_folder_path = "/mnt/disks/proto/stl10_images_rotated/"


# ----------------- Data ----------------- #


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    


class ContrastiveImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png", ".JPEG")):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


# ----------------- Experiments ----------------- #


class SimCLR(torchplate.experiment.Experiment):
    # implements the simclr method 
    def __init__(self, hidden_dim=128, temperature=0.1, n_views=2):
        # simclr hyperparams
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.n_views = n_views

        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"Using device: {self.device}")

        # model 
        # Base model f(x)
        self.model = torchvision.models.resnet18(num_classes = 4 * self.hidden_dim) 
        # f(x) -> f(x) + g(x)
        self.model.fc = nn.Sequential(
            self.model.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.model.to(self.device)


        # misc 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        
        # data
        transformations = self.get_default_transforms()
        self.ds = ContrastiveImageDataset(root_dir=data_folder_path, 
                   transform=ContrastiveTransformations(transformations, n_views=self.n_views)
                   )
        self.trainloader = DataLoader(self.ds, batch_size=batch_size, shuffle=True)    
    

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            save_weights_every_n_epochs = None, 
            wandb_logger = None,
            verbose = True,
            experiment_name = misc.timestamp()
        )


    def test_data_loader(self):
        # test the data loader 
        for batch in self.trainloader:
            loss_ = self.info_nce_loss(batch)
            print(loss_)
            break


    def get_default_transforms(self):
        default_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=96),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
        
        return default_transforms
    

    
    def info_nce_loss(self, batch):
        # implements info nce loss 
        imgs = torch.cat(batch, dim=0)
        imgs = imgs.to(self.device)
        latents = self.model(imgs)
        cos_sim = F.cosine_similarity(latents[:,None,:], latents[None,:,:], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll
    

    def evaluate(self, batch):
        loss = self.info_nce_loss(batch)
        return loss


    def on_run_end(self):
        self.save_weights()




# ----------------- Runner ----------------- #

def main(args):
    exp = SimCLR(hidden_dim=hidden_dim, temperature=temperature, n_views=n_views)
    exp.train(num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    args = parser.parse_args()
    main(args)
