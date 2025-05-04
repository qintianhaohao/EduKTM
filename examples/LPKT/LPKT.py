#!/usr/bin/env python
# coding: utf-8

import numpy as np
from load_data import DATA
from EduKTM import LPKT
import logging


def generate_q_matrix(path, n_skill, n_problem, gamma=0):
    """
    生成Q矩阵。

    参数:
    path (str): 包含问题到技能映射数据的文件路径。
    n_skill (int): 技能的数量。
    n_problem (int): 问题的数量。
    gamma (float): Q矩阵中非相关项的初始值。

    返回:
    numpy.ndarray: Q矩阵。
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1
    return q_matrix


def prepare_data(seqlen, train_path, test_path, problem2skill_path, n_question, n_exercise, q_gamma):
    """
    准备训练和测试数据。

    参数:
    seqlen (int): 数据序列长度。
    train_path (str): 训练数据文件路径。
    test_path (str): 测试数据文件路径。
    problem2skill_path (str): 问题到技能映射数据文件路径。
    n_question (int): 问题数量。
    n_exercise (int): 练习数量。
    q_gamma (float): Q矩阵中非相关项的初始值。

    返回:
    tuple: 包含训练数据、测试数据和Q矩阵的元组。
    """
    dat = DATA(seqlen=seqlen, separate_char=',')
    train_data = dat.load_data(train_path)
    test_data = dat.load_data(test_path)
    q_matrix = generate_q_matrix(problem2skill_path, n_question, n_exercise, q_gamma)
    return train_data, test_data, q_matrix


def train_model(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout, train_data, test_data,
                epoch, lr):
    """
    训练LPKT模型并保存参数。

    参数:
    各种模型参数和数据集。

    返回:
    None
    """
    lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout)
    lpkt.train(train_data, test_data, epoch=epoch, lr=lr)
    lpkt.save("lpkt.params")


def evaluate_model(param_path, test_data, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size):
    """
    加载模型参数并评估模型性能。

    参数:
    param_path (str): 模型参数文件路径。
    test_data (any): 测试数据。
    n_at (int): 参数 n_at。
    n_it (int): 参数 n_it。
    n_exercise (int): 参数 n_exercise。
    n_question (int): 参数 n_question。
    d_a (int): 参数 d_a。
    d_e (int): 参数 d_e。
    d_k (int): 参数 d_k。
    q_matrix (numpy.ndarray): Q 矩阵。
    batch_size (int): 批量大小。

    返回:
    tuple: AUC 和准确率。
    """
    lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size)
    lpkt.load(param_path)
    _, auc, accuracy = lpkt.eval(test_data)
    print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
    return auc, accuracy



if __name__ == "__main__":
    # 配置日志记录器
    logging.getLogger().setLevel(logging.INFO)

    # 参数设置
    batch_size = 32
    n_at = 1326
    n_it = 2839
    n_question = 102
    n_exercise = 3162
    seqlen = 500
    d_k = 128
    d_a = 50
    d_e = 128
    q_gamma = 0.03
    dropout = 0.2

    # 准备数据
    train_data, test_data, q_matrix = prepare_data(
        seqlen,
        '../../data/anonymized_full_release_competition_dataset/train0.txt',
        '../../data/anonymized_full_release_competition_dataset/test.txt',
        '../../data/anonymized_full_release_competition_dataset/problem2skill',
        n_question,
        n_exercise,
        q_gamma
    )

    # 训练模型
    train_model(
        n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout,
        train_data, test_data, epoch=2, lr=0.003
    )

    # 评估模型
    evaluate_model(
        "lpkt.params", test_data,
        n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size
    )

