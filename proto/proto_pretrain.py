"""
File: proto-pretrain.py
------------------
Implements prototypical pre-training. 
"""


import torchplate
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import torchvision 
from torch.utils.data import Dataset
import wandb
from torchvision import transforms
import pdb
import os
import rsbox 
from rsbox import ml, misc



# ----------------- Global config vars ----------------- #



train_folder = "/mnt/disks/proto/stl_10/train" 
batch_size = 512
logger = None
experiment_name = "stl10-baseline" + "-" + misc.timestamp()
normalize = True
resize = None
# resnet_weights = torchvision.models.ResNet18_Weights.DEFAULT
resnet_weights = None
experiment_name = "proto-pretrain" + "-" + misc.timestamp()
latent_dim = 128 * 4
output_dim = 128
proto_method = 'query'
print("batch size: ", batch_size)
print("latent dim: ", latent_dim)
print("output dim: ", output_dim)
print("resnet weights: ", resnet_weights)
print("normalization: ", normalize)
print("proto method: ", proto_method)


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


class ProtoEncoder(nn.Module):
    """
    Helper class which functionizes two separate PyTorch modules
    (encoder + projection head). Inherits from torch.nn.Module so 
    instantiations of this class are trainable PyTorch modules. 
    Functionally, pre-train the encoder_module and then fine-tune
    this linear probe. 
    This one uses a resnet18 for the encoder function. 
    """

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        
        # hyperparameters 
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Base encoder model f(x)
        # self.encoder = torchvision.models.resnet18(num_classes = self.latent_dim, weights=resnet_weights)
        self.encoder = torchvision.models.resnet18(num_classes = self.latent_dim)

        # add projection head g(x) 
        self.encoder.fc = nn.Sequential(
            self.encoder.fc, 
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.output_dim)
        )


    def forward(self, x):
        return self.encoder(x)



# ----------------- Training Experiments ----------------- #



class ProtoPretrainExperiment(torchplate.experiment.Experiment):
    """
    Our implementation of our proposed approach: 
    prototypical pre-training. In this module,
    we learn the weights for the pre-trained
    encoder (i.e. performs the pre-training). 
    """
    def __init__(self):

        print(f"Running ProtoPretrainExperiment. Run name: {experiment_name}")

        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"Using device: {self.device}")

        # model 
        self.model = ProtoEncoder(latent_dim, output_dim)
        self.model.to(self.device)

        # misc 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


        # data 
        self.transforms = None
        self.trainset = ImageClassificationDataset(train_folder, transform=self.transforms)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        print(f"train set len ~{len(self.trainloader) * batch_size}")

        self.prev_loss = -2.0
        self.prev_acc = -2.0

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True
        )


    def create_class_vectors(self, x, y):
        """
        Helper function. Creates a dict of vectors for each class. 
        Of the form: 
        {
            class_1: [latent_vector_1, latent_vector_2, ...],
            class_2: [latent_vector_1, latent_vector_2, ...],
            ...
        }
        """

        # get the labels 

        class_vectors = {}

        for i in range(len(y)):
            key_ = str(int(y[i]))
            if key_ not in class_vectors:
                class_vectors[key_] = []
            
            class_vectors[key_].append(x[i])


        return class_vectors
    

    def create_label_map(self, labels):
        """
        Helper function. Creates a mapping from label to index for CE loss. 
        """
        unique_labels = torch.unique(labels)
        
        label_map = {}

        for i, elem in enumerate(unique_labels):
            label_map[str(int(elem))] = i

        # e.g., {'1': 0, '2': 1, '3': 2}

        return label_map
    

    def transform_labels(self, labels, label_map):
        """
        Applys the label map to the labels. 
        """
        new_labels = []

        for label in labels:
            new_labels.append(label_map[str(int(label))])

        return torch.tensor(new_labels)
    

    def calculate_score(self, logits, labels):
        """Calculates accuracy given logits and labels.
        Source: CS 330 Assignment 2. 
        Args:
            logits (Tensor): shape (N, C)
            y (Tensor): shape (N,)
        Returns:
            accuracy (float)
        """
        assert logits.dim() == 2
        assert labels.dim() == 1
        assert logits.shape[0] == labels.shape[0]
        y = torch.argmax(logits, dim=-1) == labels
        y = y.type(torch.float)
        return torch.mean(y).item()
    
    
    def proto_step(self, xs, ys):
        """
        Runs the protonet approach to produce the logits for
        the given input for the given task. 
        """

        # embed into projection space (will be of shape [batch_size, output_dim])        
        embeddings = self.model(xs)  # e.g. (16, 128)

        # compute class vectors
        label_map = self.create_label_map(ys)
        new_labels = self.transform_labels(ys, label_map)
        class_vectors = self.create_class_vectors(embeddings, new_labels)
        

        # get mean for each class (from support)
        prototypes = {}
        for key in class_vectors:
            prototypes[key] = torch.mean(torch.stack(class_vectors[key]), dim=0)

        proto_list = [prototypes[key] for key in sorted(prototypes.keys())]
        protos = torch.stack(proto_list)

        # ^ gets protos [0,...,num_unique_labels] 

        logits = -torch.cdist(embeddings, protos)  # questionable 
        new_labels = new_labels.to(self.device)
        loss_val = F.cross_entropy(logits, new_labels)
        accuracy_val = self.calculate_score(logits, new_labels)

        return loss_val, accuracy_val


    def proto_step_query(self, xs, ys):
        """
        Version number 2: divide the batch into 1/4 query, 3/4 support. 
        Calculate the prototypes only based on support set. Loss is
        based on query set to encourage generalization. 
        """

        # construct support set to be 3/4 of the data points and query set to be 1/4
        query_size = len(ys) // 4
        support_size = len(ys) - query_size
        assert query_size + support_size == len(ys)
        xs_support = xs[:support_size]
        ys_support = ys[:support_size]
        xs_query = xs[support_size:]
        ys_query = ys[support_size:]
        assert len(xs_support) == support_size and len(xs_query) == query_size

        # need to assert that all of the query labels exist in the support labels
        for label in ys_query:
            if label not in ys_support:
                print('neg one...')
                return -1, -1


        # embed into projection space (will be of shape [batch_size, output_dim])        
        embeddings = self.model(xs_support)  # e.g. (16, 128)

        # compute class vectors
        label_map = self.create_label_map(ys_support)
        new_labels = self.transform_labels(ys_support, label_map)
        class_vectors = self.create_class_vectors(embeddings, new_labels)
        

        # get mean for each class (from support)
        prototypes = {}
        for key in class_vectors:
            prototypes[key] = torch.mean(torch.stack(class_vectors[key]), dim=0)

        proto_list = [prototypes[key] for key in sorted(prototypes.keys())]
        protos = torch.stack(proto_list)


        # query calculations 
        query_latents = self.model(xs_query)
        new_query_labels = self.transform_labels(ys_query, label_map)
        new_query_labels = new_query_labels.to(self.device)

        logits = -torch.cdist(query_latents, protos)
        loss_val = F.cross_entropy(logits, new_query_labels)
        accuracy_val = self.calculate_score(logits, new_query_labels)

        return loss_val, accuracy_val
    


    def evaluate(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        if proto_method == "standard":
            loss_, acc_ = self.proto_step(x, y)
        elif proto_method == "query":
            loss_, acc_ = self.proto_step_query(x, y)
        else:
            loss_, acc_ = self.proto_step(x, y)

        self.prev_loss = loss_
        self.prev_acc = acc_

        metrics_dict = {'loss': loss_, 'accuracy': acc_}

        return metrics_dict


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
    exp = ProtoPretrainExperiment()
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?')
    args = parser.parse_args()
    main(args)