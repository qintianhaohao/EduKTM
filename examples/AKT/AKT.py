#!/usr/bin/env python
# coding: utf-8

"""
Attentive Knowledge Tracing (AKT) 模型的实现和使用示例。
该脚本展示了如何准备数据、训练模型、保存模型参数以及加载模型进行测试。
"""

import logging
from load_data import DATA, PID_DATA
from EduKTM import AKT


def prepare_data(model_type, n_question, seqlen, train_file_path, test_file_path):
    """
    数据准备函数。
    加载并处理训练和测试数据集。

    参数:
        model_type (str): 模型类型，例如 'pid'。
        n_question (int): 问题数量。
        seqlen (int): 序列长度。
        train_file_path (str): 训练数据文件路径。
        test_file_path (str): 测试数据文件路径。

    返回值:
        tuple: 包含训练数据和测试数据的元组。
    """
    if model_type == 'pid':
        dat = PID_DATA(n_question=n_question, seqlen=seqlen, separate_char=',')
    else:
        dat = DATA(n_question=n_question, seqlen=seqlen, separate_char=',')

    train_data = dat.load_data(train_file_path)
    test_data = dat.load_data(test_file_path)
    return train_data, test_data


def train_and_save_model(n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2,
                         batch_size, maxgradnorm, train_data, test_data, model_path):
    """
    模型训练与保存函数。
    初始化模型、训练模型并将模型参数保存到指定路径。

    参数:
        n_question (int): 问题数量。
        n_pid (int): 学生问题 ID 的数量。
        n_blocks (int): Transformer 块的数量。
        d_model (int): 模型维度。
        dropout (float): Dropout 比率。
        kq_same (int): 是否共享 Key 和 Query 的标志。
        l2 (float): L2 正则化系数。
        batch_size (int): 批次大小。
        maxgradnorm (float): 最大梯度范数。
        train_data: 训练数据。
        test_data: 测试数据。
        model_path (str): 模型参数保存路径。
    """
    # 初始化模型
    akt = AKT(n_question=n_question, n_pid=n_pid, n_blocks=n_blocks, d_model=d_model,
              dropout=dropout, kq_same=kq_same, l2=l2, batch_size=batch_size, maxgradnorm=maxgradnorm)

    # 训练模型
    logging.getLogger().setLevel(logging.INFO)
    akt.train(train_data, test_data, epoch=2)

    # 保存模型参数
    akt.save(model_path)


def load_and_test_model(model_path, test_data, n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2, batch_size,
                        maxgradnorm):
    """
    模型加载与测试函数。
    从指定路径加载模型参数并在测试数据上评估模型性能。

    参数:
        model_path (str): 模型参数文件路径。
        test_data: 测试数据。
        n_question (int): 问题数量。
        n_pid (int): 学生问题 ID 的数量。
        n_blocks (int): Transformer 块的数量。
        d_model (int): 模型维度。
        dropout (float): Dropout 比率。
        kq_same (int): 是否共享 Key 和 Query 的标志。
        l2 (float): L2 正则化系数。
        batch_size (int): 批次大小。
        maxgradnorm (float): 最大梯度范数。

    返回值:
        tuple: 包含 AUC 和准确率的元组。
    """
    # 初始化模型（使用与训练时相同的参数）
    akt = AKT(n_question=n_question, n_pid=n_pid, n_blocks=n_blocks, d_model=d_model,
              dropout=dropout, kq_same=kq_same, l2=l2, batch_size=batch_size, maxgradnorm=maxgradnorm)

    # 加载模型参数
    akt.load(model_path)

    # 测试模型
    _, auc, accuracy = akt.eval(test_data)
    print("AUC: %.6f, 准确率: %.6f" % (auc, accuracy))
    return auc, accuracy



if __name__ == "__main__":
    # 配置参数
    batch_size = 64
    model_type = 'pid'
    n_question = 123
    n_pid = 17751
    seqlen = 200
    n_blocks = 1
    d_model = 256
    dropout = 0.05
    kq_same = 1
    l2 = 1e-5
    maxgradnorm = -1

    # 数据路径
    train_file_path = '../../data/2009_skill_builder_data_corrected/train_pid.txt'
    test_file_path = '../../data/2009_skill_builder_data_corrected/test_pid.txt'

    # 模型参数保存路径
    model_path = "akt.params"

    # 数据准备
    train_data, test_data = prepare_data(model_type, n_question, seqlen, train_file_path, test_file_path)
    print('数据准备成功！')

    # 训练并保存模型
    train_and_save_model(n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2,
                         batch_size, maxgradnorm, train_data, test_data, model_path)
    print('训练完成！')

    # 加载模型并测试
    load_and_test_model(model_path, test_data, n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2, batch_size,
                        maxgradnorm)
    print('测试完成！')