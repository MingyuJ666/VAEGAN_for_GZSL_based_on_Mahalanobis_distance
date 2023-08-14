import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import os
import re
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Model import Encoder, Decoder, DiscriminatorA, DiscriminatorB
from torch.utils.data import Dataset
from PIL import Image
from utils import get_class, recover
import argparse
from torch.autograd import Variable
from Matrix import get_mahalanobis
import pandas as pd
import csv

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
parser.add_argument('--root_path', type=str, default='...',
                    help="The root path of dataset.")
parser.add_argument('--test_path', type=str, default='.../test/set4',
                    help="The test path of dataset.")
opt = parser.parse_args()


device = torch.device(opt.device)

DiscriminatorB = DiscriminatorB(opt.image_size)

discriminatorB_weights_path = 'discriminatorB.pth'

DiscriminatorB.load_state_dict(torch.load(discriminatorB_weights_path))


maha_matrix = torch.load('lossmat_nc.pt')
mat_m_t = torch.load("mat_M_t.pt")

# transforms
transform_test = transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Data loading
test_dataset = datasets.ImageFolder(root=opt.test_path, transform=transform_test)
train_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
print("images loaded!")
classname = get_class(opt.root_path)


import glob
def get_class1(dataset_root):
    class_dirs = os.path.join(dataset_root)
    class_names = []
    sub_folders = sorted([folder for folder in os.listdir(class_dirs)
                          if os.path.isdir(os.path.join(class_dirs, folder))])
    for folder in sub_folders:
        folder_path = os.path.join(class_dirs, folder)
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        for image_path in image_paths:
            filename = image_path.split("/")[-1]
            class_name = re.sub(r'\d+', '', filename)
            class_name = class_name.strip('_')
            name = class_name.split("__", 1)[0]
            class_names.append(name)

    return class_names

testname = get_class1(opt.test_path)

seen_class = mat_m_t

cnt = 0
q = 0
sum = 0
for i, (images, _) in enumerate(train_loader):
    imgT = Variable(images)  # .unsqueeze(0)
    ds_real = DiscriminatorB(imgT)
    list_cp = []
    img_dict = {}
    maha_dict = {}

    for i in range(len(seen_class[0])):
        seen_vec = seen_class[:, i].view(-1, 1)
        loss = get_mahalanobis(ds_real, seen_vec)
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
    top_values, top_indices = torch.topk(L_tensor, k=5, largest=True)

    print("Prediction Rank: \n")
    print('Class Name:', testname[q])

    classa = ''
    print(top_indices.flatten() + 1)
    p = 0
    for i in top_indices.flatten():
        if p==0:
            classa = classname[i]

        p+=1
        classe = classname[i]
        print(classe)


    if(classa == testname[q]):
        cnt = cnt + 1
        print("\n")
        print("Result: Correct!")
        print("\n")
        print(classa)
    else:
        print("\n")
        print("Result: Wrong!")
        print("\n")
    sum+=1
    q += 1



acc = cnt/sum * 100
print("Summery test image numbers：", sum)
print("Accuracy: {:.2f}%".format(acc))



















#     img_dict = {}
#     maha_dict = {}
#     list_cp = []
#     dictu = {}
#     list1 = []
#     print(labels)
#     print(seen_class.shape)
#     for i in range(len(classname)):
#         seen_vec = seen_class[:, i].view(-1, 1)
#         dis = ds_real - seen_vec
#         dis_abs_sum = torch.abs(dis).sum()
#         img_dict[i] = dis_abs_sum
#
#     for i in range(len(classname)):
#         seen_vec = seen_class[:, i].view(-1, 1)
#         loss = ds_real - seen_vec
#         abs_sum = torch.abs(loss).sum()
#         list_cp.append(abs_sum)
#
#     tensor_list = torch.tensor(list_cp)
#     tensor_list = tensor_list.view(len(list_cp), 1)
#     maha_vec = maha_matrix[:, 1]
#
#     for i in range(len(list_cp)):
#         maha_dict[i] = maha_vec[i]
#
#     sort_rank = [k for k, v in sorted(maha_dict.items(), key=lambda item: item[1], reverse=False)]
#
#
#     for i in range(maha_matrix.shape[1]):
#         seen_vec = maha_matrix[:, i].view(-1, 1)
#         loss1 = get_mahalanobis(tensor_list, seen_vec)
#         abs_sum = torch.abs(seen_vec).sum()
#         dictu[i] = abs_sum
#         list1.append(abs_sum)
#
#     tensor_lst1 = torch.tensor(list1)
#     sorted_keys = [k for k, v in sorted(dictu.items(), key=lambda item: item[1], reverse=False)]
#     L_tensor = torch.mm(ds_real.t(), seen_class)
#     top_values, top_indices = torch.topk(L_tensor, k=3, largest=True)
#     classe_list = []
#     print("Prediction Rank in top 3: \n")
#     # print(top_indices.flatten() + 1)
#     for i in top_indices.flatten():
#         classe = classname[i]
#         classe_list.append(classe)
#         print(classe)
#     #label process
#     labels = str(labels)
#     pattern = r"'(.+)'"
#     match = re.search(pattern, labels)
#     result = match.group(1) if match else None
#     if(classe_list[0] == result):
#         cnt = cnt + 1
#         print("\n")
#         print("Result: Correct!")
#         print("\n")
#         print(result)
#     else:
#         print("\n")
#         print("Result: Wrong!")
#         print("\n")
#         print(result)

