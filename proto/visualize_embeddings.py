"""
File: visualize_embeddings.py
------------------
Script to visualize the embeddings produced by the pre-trained encoder
by class label using t-SNE. 
"""


import rsbox 
from rsbox import ml, misc 
import numpy as np 
import torch 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import proto_pretrain 
from proto_pretrain import ProtoEncoder
import torchvision 
from torch.utils.data import Dataset


# ----------------- Global config vars ----------------- #


# ds_path = "/mnt/disks/proto/stl10_tr_viz"
# ds_path = "/mnt/disks/proto/stl_10/train" 
ds_path = "/mnt/disks/proto/stl_10/test"
resize = None
normalize = True
extension = 'png'
plot_title = "stl-10-PRL-query-test-epoch"
dir_path = "plots"
file_save_name = plot_title
# file_save_name = "stl-10-PRL-query-mini-train"
latent_dim = 128 * 4
output_dim = 128
freeze = True
num_classes = 10
# pretrained_encoder_path = "saved/11-10-AM-Jun-08-2023.pth"
pretrained_encoder_path = "saved/1-13-PM-Jun-08-2023.pth"  # query method weights 


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




# ----------------- Helper functions ----------------- #


def encode_features(ds, encoder):
    encoded_features = []
    labels = []

    with torch.no_grad():
        for x, y in ds:
            x = torch.unsqueeze(torch.tensor(x), 0)
            encoded_x = encoder.encode(x)
            encoded_features.append(encoded_x)
            labels.append(y)

    encoded_features = torch.cat(encoded_features, dim=0).numpy()
    labels = np.array(labels)

    return encoded_features, labels



# ----------------- Runner ----------------- #


ds = ml.classification_dataset(ds_path, resize=resize, normalize=normalize, extension=extension)
encoder = LinearProbe(
            latent_dim, 
            output_dim, 
            num_classes, 
            pretrained_encoder_path=pretrained_encoder_path, 
            freeze=freeze
        ).double()
encoded_features, labels = encode_features(ds, encoder)



tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(encoded_features)



num_classes = len(np.unique(labels))
color_map = plt.cm.get_cmap('viridis', num_classes)
fig, ax = plt.subplots()
sc = ax.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap=color_map)
handles, labels_legend = sc.legend_elements()
legend = ax.legend(handles, labels_legend, loc='best', title='Classes')
plt.title(f'{plot_title} t-SNE')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.box(on=None)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
save_path_f = dir_path + "/" + file_save_name + ".png"
plt.savefig(save_path_f)
print("Saved t-SNE plot to: ", save_path_f)


plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap=color_map)
plt.colorbar()
plt.show()