import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision.transforms as transforms
from PIL import Image
from Model import Encoder, Decoder, DiscriminatorA, DiscriminatorB
from torch.autograd import Variable
import torchvision.datasets as datasets
from utils import generate_binary_codes
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--Test_image_path', type=str, default='',
                    help='The path of Test images dataset.')
parser.add_argument('--image_size', type=int, default='256',  help='The input image size.')
parser.add_argument('--image_path', type=str, default='',
                    help='The path of image dataset.')
parser.add_argument('--text_path', type=str, default='',
                    help='The path of text dataset.')

opt = parser.parse_args()


transform = transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
Binary_code_list = generate_binary_codes(opt.image_path)

train_dataset = datasets.ImageFolder(root=opt.image_path,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
print("images loaded!")