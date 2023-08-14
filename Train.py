import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision
import argparse
from Model import Encoder, Decoder, DiscriminatorA, DiscriminatorB
from Matrix import vec_to_mat, Maha_loss, get_mahalanobis, get_mahalanobis2, get_loss_func, reshape_vector
from utils import generate_binary_codes, get_semantic, recover
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from transformers import logging
import csv
import gc
import os
import time

logging.set_verbosity_error()
parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default='256', help='The input image size.')
parser.add_argument('--h_dim', type=int, default='512', help='The h_dim.')
parser.add_argument('--z_dim', type=int, default='500', help='The z_dim.')
parser.add_argument('--num_epochs', type=int, default='5', help='The number of epochs.')
parser.add_argument('--batch_size', type=int, default='1', help='The number of images in 1 batch')
parser.add_argument('--learning_rate', type=float, default='4e-5', help='The learning rate of vaegan.')
parser.add_argument('--device', type=str, default='mps:0', help='The running device.')
parser.add_argument('--device_alt', type=str, default='cpu', help='The alternative device.')
parser.add_argument('--bert_name', type=str, default="bert-base-uncased", help='The semantic model name.')
parser.add_argument('--root_path', type=str, default='',help="The root path of dataset.")

opt = parser.parse_args()

h_dim = opt.h_dim
z_dim = opt.z_dim
image_size = opt.image_size
image_path = opt.root_path + 'images'
# Images transforms
transform = transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Data loading
train_dataset = datasets.ImageFolder(root=image_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

device = torch.device(opt.device if torch.cuda.is_available() else opt.device_alt)


# generate binary codes
Binary_code_list = generate_binary_codes(opt.root_path + 'images')

# load pretrained model: resnet101, Bert
tokenizer = BertTokenizer.from_pretrained(opt.bert_name)
bertmodel = BertModel.from_pretrained(opt.bert_name)
resnet101 = models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1).to(device)
# resnet101 = models.resnet101(pretrained = True).to(device)  # old version

# Instantiating Classes
encoder = Encoder(opt.h_dim, opt.z_dim).to(device)
decoder = Decoder(opt.h_dim, opt.z_dim).to(device)
discriminatorA = DiscriminatorA(opt.image_size).to(device)

discriminatorB = DiscriminatorB(opt.image_size).to(device)
params = list(encoder.parameters()) + list(decoder.parameters())


# Optimizer
optimizer_ge = optim.Adam(params, lr=opt.learning_rate)
optimizer_d = optim.Adam(discriminatorA.parameters(), lr=opt.learning_rate)
optimizer_db = optim.Adam(discriminatorB.parameters(), lr=opt.learning_rate)

# Loss functions
reconstruction_loss = nn.MSELoss()
adversarial_loss = nn.BCELoss()

# Bert sematic process
list_sem = get_semantic(opt.root_path + 'text')

if os.path.exists('train_mat_y.csv') and os.path.exists('train_mat_t.csv'):
    os.remove('train_mat_y.csv')
    os.remove('train_mat_t.csv')
    print("File deleted successfully!")
else:
    print("File does not exist.")


# Training
for epoch in range(opt.num_epochs):

    for i, (images, _) in enumerate(train_loader):
        total_iterations = len(train_loader)
        images = Variable(images).to(device)
        with torch.no_grad():
            image = resnet101(images)

        text = list_sem[i]
        tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
        outputs = bertmodel(**tokens)
        last_hidden_state = outputs.last_hidden_state
        text_encoding = last_hidden_state[:, 0, :]
        # (1,768)

        concatenated_vector = torch.cat((text_encoding, image), dim=1)
        output_tensor = reshape_vector(concatenated_vector)

        z, _ = encoder(output_tensor)
        x_reconstructed = decoder(z)
        d_fake = discriminatorA(x_reconstructed)

        optimizer_d.zero_grad()
        d_real = discriminatorB(images)
        tensor_f = d_fake.squeeze().tolist()
        tensor_r = d_real.squeeze().tolist()

        if epoch == (opt.num_epochs - 1):
            with open('train_mat_y.csv', mode='a', newline='') as file1:
                writer = csv.writer(file1)
                vec_to_mat(tensor_f, writer)
            time.sleep(0.5)
        if epoch == (opt.num_epochs - 1):
            with open('train_mat_t.csv', mode='a', newline='') as file2:
                writer = csv.writer(file2)
                vec_to_mat(tensor_r, writer)
            time.sleep(0.5)

        loss_real = adversarial_loss(d_real, Variable(torch.ones(900, 1)).to(device))
        loss_fake = adversarial_loss(d_fake, Variable(torch.zeros(900, 1)).to(device))
        loss_d = loss_real + loss_fake
        loss_d.backward(retain_graph=True)
        optimizer_d.step()


        # loss_g
        optimizer_ge.zero_grad()
        loss_g = reconstruction_loss(x_reconstructed, images)
        loss_g.backward(retain_graph=True)

        # loss on maha loss
        optimizer_db.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        # loss_m_val = get_mahalanobis(d_fake, d_real)
        loss_m_val = get_loss_func(d_fake.t(), d_real.t())
        loss_m = torch.tensor(loss_m_val, requires_grad=True)
        loss_m.backward(retain_graph=True)
        optimizer_ge.step()
        optimizer_db.step()

        if (i + 1) % 1 == 0:
            print(
                "**********************************************************************************************")
            print(f"Epoch [{epoch + 1}/{opt.num_epochs}], Step [{i + 1}/{len(train_loader)}], "
                  f"D_loss: {loss_d.item():.4f}, G_loss: {loss_g.item():.4f}, M_loss: {loss_m.item():.4f}")
            print("**********************************************************************************************")


# Save model parameters
torch.save(discriminatorB.state_dict(), 'discriminatorB.pth')
print("Training Done!")
