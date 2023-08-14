import math
import numpy as np
import torch


def vec_to_mat1(mat_y, mat_t, d_fake, d_real):
    arr_fake = d_fake.t()
    arr_real = d_real.t()
    mat_y = torch.cat((mat_y, arr_fake), dim=0)
    mat_t = torch.cat((mat_t, arr_real), dim=0)
    print("mat_y", mat_y.shape)
    print("mat_t", mat_t.shape)

    return mat_y, mat_t


def vec_to_mat(d_vec, writer):
    tensor_list = d_vec
    writer.writerow(tensor_list)
    print('save successfuly')
    return 1


def Maha_loss(train_mat_y, train_mat_t, test_mat_y, test_mat_t):
    tensor_seen = train_mat_y - train_mat_t
    tensor_unseen = test_mat_y - test_mat_t
    tensor_seen_t = tensor_seen.t()
    tensor_unseen_t = tensor_unseen.t()
    print(tensor_seen.shape)
    print(tensor_unseen.shape)
    correlation_M = torch.corrcoef(tensor_seen_t)
    correlation_M = correlation_M + 1e-5
    inv_correlation_M = torch.inverse(correlation_M)
    Train_nc_p1 = torch.mm(tensor_seen, inv_correlation_M)
    Train_nc = torch.mm(Train_nc_p1, tensor_unseen_t)

    return Train_nc


def calculate_l2_norm(matrix):
    squared_matrix = torch.square(matrix)  # ^2
    sum_squared = torch.sum(squared_matrix)  # sum
    l2_norm = torch.sqrt(sum_squared)  # sqrt

    return l2_norm


def cov_mat(tensor_x, tensor_y):
    stacked_tensor = torch.stack([tensor_x, tensor_y])
    cov_matrix = torch.cov(stacked_tensor)

    return cov_matrix


def get_mahalanobis(tensor_x, tensor_y):
    mean_cat = torch.cat([tensor_x, tensor_y], dim=0).mean(dim=0, keepdim=True)
    cov_cat = torch.cat([tensor_x - mean_cat, tensor_y - mean_cat], dim=0)
    inv_cov_m = torch.inverse(
        torch.matmul(cov_cat.t(), cov_cat) / (cov_cat.size(0) - 1) + 1e-7
    )
    cov_cat_t = cov_cat.t()
    diff = cov_cat_t[..., None] - cov_cat_t[:, None, :]
    distance = torch.sqrt(torch.einsum("ijk,kl,ijl->ij", diff, inv_cov_m, diff))
    result = torch.sqrt(torch.sum(distance ** 2)) + 0.5

    return result


# def get_mahalanobis2(tensor_x, tensor_y):
#     diff_vec = tensor_x - tensor_y
#     diff_vec_t = diff_vec.t()
#     mean = tensor_x + tensor_y
#     mean = mean/2
#     tensor_cat = torch.cat([tensor_x, tensor_y], dim=0)
#     # print("tensor_cat", tensor_cat.shape)
#     correlation_M = torch.cov(tensor_cat.t())
#     inv_correlation_M = torch.inverse(correlation_M)
#     a = (tensor_x - mean).t()
#     b = tensor_x - mean
#     product1 = torch.mm(a.t(), inv_correlation_M)
#     product2 = torch.mm(product1, b.t())
#     print("product2",product2)
#     distance = torch.sqrt(product2)
#
#     return distance

def compute_cov(X):
    # print(X)
    cov_M = torch.cov(X.t())
    # print('cov',cov_M)
    ret_M = torch.linalg.pinv(cov_M) + 1e-6
    # print(ret_M)
    return ret_M


def get_mahalanobis2(tensor_x, tensor_y, M):
    diff = tensor_x - tensor_y
    distance = torch.mm(torch.mm(diff, M), diff.t())
    distance = torch.sqrt(distance)
    return distance


# X1: fake output
# X2: real output
def get_loss_func(X1, X2):
    # cat the outputs of two branches
    X = torch.cat([X1, X2], dim=0)
    print(X.shape)
    M = compute_cov(X)
    m, d = X1.shape
    loss = 0
    for i in range(m):
        for j in range(i + 1, m):
            # fake and real
            dist1 = get_mahalanobis2(X1[i, :], X2[j, :], M)
            # fake and fake
            dist2 = get_mahalanobis2(X1[i, :], X1[j, :], M)
            loss += (dist1 - dist2)

    return - math.log(loss + 1e-10)


def get_euclidean(tensor_x, tensor_y):
    diff = tensor_x - tensor_y
    squared_diff = diff.pow(2)
    sum_squared_diff = torch.sum(squared_diff)
    euclidean_dist = torch.sqrt(sum_squared_diff)

    return euclidean_dist

def reshape_vector(input_vector):
    # 检查输入向量维度是否正确
    assert input_vector.shape == (1, 1768), "输入向量维度应为 (1, 1768)"

    # 创建目标形状的张量
    reshaped_vector = torch.zeros(1, 3, 256, 256)

    # 将向量的元素按顺序填充到张量中
    for i in range(1768):
        channel = i % 3
        row = (i // 3) // 256
        col = (i // 3) % 256
        reshaped_vector[0, channel, row, col] = input_vector[0, i]

    return reshaped_vector