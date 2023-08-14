import torch
import csv
import re
from Matrix import vec_to_mat, Maha_loss
from Matrix import Maha_loss, get_mahalanobis


def recover2(mat_path):
    data = []
    with open(mat_path, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            u = []
            for i in row:
                u.append(float(i))
            data.append(u)
    # r = []
    # for i in range(len(data) - 1, -1, -1):
    #     r.append(data[i])

    data = torch.tensor(data)

    print(data.shape)
    return data


train_mat_y = recover2('train_mat_y.csv') # discriminator a






















unseen_vec = train_mat_y[300, :].unsqueeze(-1)
img_dict = {}
for i in range(len(train_mat_y)):
    seen_vec = train_mat_y[i, :].unsqueeze(-1)
    dis = seen_vec - unseen_vec
    result = torch.abs(dis.sum())

    img_dict[i] = result



sort_rank = dict(sorted(img_dict.items(), key=lambda item: item[1]))
print(sort_rank)


