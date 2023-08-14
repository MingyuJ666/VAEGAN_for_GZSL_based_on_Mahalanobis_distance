import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision.transforms as transforms
from PIL import Image
from Model import Encoder, Decoder, DiscriminatorA, DiscriminatorB
from torch.autograd import Variable
from utils import generate_binary_codes, get_class
import argparse
from Matrix import Maha_loss, get_mahalanobis

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default='256', help='The input image size.')
parser.add_argument('--h_dim', type=int, default='512', help='The h_dim.')
parser.add_argument('--z_dim', type=int, default='500', help='The z_dim.')
parser.add_argument('--num_epochs', type=int, default='10', help='The number of epoches.')
parser.add_argument('--batch_size', type=int, default='1', help='The number of images in 1 batch')
parser.add_argument('--learning_rate', type=float, default='5e-3', help='The learning rate of vaegan.')
parser.add_argument('--bert_name', type=str, default="bert-base-uncased", help='The semantic model name.')
parser.add_argument('--device', type=str, default='cpu', help='The running device.')
parser.add_argument('--device_alt', type=str, default='cuda', help='The alternative device.')
parser.add_argument('--image_path', type=str, default='',
                    help='The path of image dataset.')
parser.add_argument('--text_path', type=str, default='',
                    help='The path of text dataset.')
parser.add_argument('--root_path', type=str, default='',
                    help="The root path of dataset.")

opt = parser.parse_args()

image_path = ''
img_ori = Image.open(image_path)
maha_matrix = torch.load("lossmat_nc.pt")
mat_m_t = torch.load("mat_M_t.pt")

device = torch.device(opt.device)
Encoder = Encoder(opt.h_dim, opt.z_dim)
Decoder = Decoder(opt.h_dim, opt.z_dim)
DiscriminatorA = DiscriminatorA(opt.image_size)
DiscriminatorB = DiscriminatorB(opt.image_size)

encoder_weight_path = 'encoder.pth'
decoder_weights_path = 'decoder.pth'
discriminatorA_weights_path = 'discriminatorA.pth'
discriminatorB_weights_path = 'discriminatorB.pth'

Encoder.load_state_dict(torch.load(encoder_weight_path))
Decoder.load_state_dict(torch.load(decoder_weights_path))
DiscriminatorA.load_state_dict(torch.load(discriminatorA_weights_path))
DiscriminatorB.load_state_dict(torch.load(discriminatorB_weights_path))

Binary_code_list = generate_binary_codes(opt.image_path)
print("Binary Len:", len(Binary_code_list))
classname = get_class(opt.root_path)
print("Class len:", len(classname))

transform = transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# input image trans to tensor
img_ori = transform(img_ori)
print("img_ori", img_ori.shape)
imgT = Variable(img_ori).unsqueeze(0)
ds_real = DiscriminatorB(imgT)

img_dict = {}
maha_dict = {}
list_cp = []
seen_class = mat_m_t

for i in range(len(classname)):
    seen_vec = seen_class[:, i].view(-1, 1)
    dis = ds_real - seen_vec
    dis_abs_sum = torch.abs(dis).sum()
    img_dict[i] = dis_abs_sum

for i in range(len(classname)):
    seen_vec = seen_class[:, i].view(-1, 1)
    loss = ds_real - seen_vec
    abs_sum = torch.abs(loss).sum()
    list_cp.append(abs_sum)

tensor_list = torch.tensor(list_cp)
tensor_list = tensor_list.view(len(list_cp), 1)
maha_vec = maha_matrix[:, 1]

for i in range(len(list_cp)):
    maha_dict[i] = maha_vec[i]

sort_rank = [k for k, v in sorted(maha_dict.items(), key=lambda item: item[1], reverse=False)]

print('list', sort_rank)

dictu = {}
list1 = []

for i in range(maha_matrix.shape[1]):
    seen_vec = maha_matrix[:, i].view(-1, 1)
    loss1 = get_mahalanobis(tensor_list, seen_vec)
    abs_sum = torch.abs(seen_vec).sum()
    dictu[i] = abs_sum
    list1.append(abs_sum)

tensor_lst1 = torch.tensor(list1)
sorted_keys = [k for k, v in sorted(dictu.items(), key=lambda item: item[1], reverse=False)]
L_tensor = torch.mm(ds_real.t(), seen_class)
top_values, top_indices = torch.topk(L_tensor, k=10, largest=True)

print("Prediction Rank: \n")
print(top_indices.flatten() + 1)
for i in top_indices.flatten():
    classe = classname[i]
    print(classe)
