#!/usr/bin/env python
# coding: utf-8

"""
该脚本包含用于处理 ASSISTments 数据集的操作，
包括数据预处理、序列解析、数据分割等。
"""

import os
import pandas as pd
import numpy as np
from EduData import get_data
from sklearn.model_selection import train_test_split, KFold
import tqdm


def download_dataset_if_needed():
    """
    如果本地不存在指定的数据集文件，则从远程下载并保存到本地。
    """
    if not os.path.exists(
            '../../data/anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv'):
        print("本地未找到数据集，正在下载...")
        get_data("assistment-2017", "../../data")
        print("数据集下载完成！")


def load_and_preprocess_data():
    """
    加载数据集并进行预处理，包括去除空值、排序和类型转换。

    返回值:
        pd.DataFrame: 预处理后的数据集。
    """
    data = pd.read_csv(
        '../../data/anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv',
        usecols=['startTime', 'timeTaken', 'studentId', 'skill', 'problemId', 'correct']
    ).dropna(subset=['skill', 'problemId']).sort_values('startTime')

    # 将 timeTaken 列转换为整数类型
    data.timeTaken = data.timeTaken.astype(int)
    return data


def generate_id_mappings(data):
    """
    生成技能、问题和答题时间的 ID 映射表。

    参数:
        data (pd.DataFrame): 数据集。

    返回值:
        tuple: 包含 skill2id, problem2id, at2id 的元组。
    """
    skills = data.skill.unique().tolist()
    problems = data.problemId.unique().tolist()
    at = data.timeTaken.unique()

    skill2id = {p: i + 1 for i, p in enumerate(skills)}
    problem2id = {p: i + 1 for i, p in enumerate(problems)}
    at2id = {a: i for i, a in enumerate(at)}

    print("技能数量: %d" % len(skills))
    print("问题数量: %d" % len(problems))
    print("答题时间数量: %d" % len(at))

    return skill2id, problem2id, at2id


def calculate_interval_time(data):
    """
    计算学生之间的间隔时间，并生成间隔时间的 ID 映射表。

    参数:
        data (pd.DataFrame): 数据集。

    返回值:
        dict: 包含间隔时间 ID 映射的字典。
    """
    it = set()
    for u in data.studentId.unique():
        startTime = np.array(data[data.studentId == u].startTime)
        for i in range(1, len(startTime)):
            item = (startTime[i] - startTime[i - 1]) // 60
            if item > 43200:
                item = 43200
            it.add(item)

    it2id = {a: i for i, a in enumerate(it)}
    print("间隔时间数量: %d" % len(it))
    return it2id


def map_problem_to_skill(data, skill2id, problem2id):
    """
    建立问题到技能的映射关系，并将结果保存到文件。

    参数:
        data (pd.DataFrame): 数据集。
        skill2id (dict): 技能 ID 映射表。
        problem2id (dict): 问题 ID 映射表。
    """
    problem2skill = {}
    for s, p in zip(np.array(data.skill), np.array(data.problemId)):
        problem2skill[problem2id[p]] = skill2id[s]

    with open('../../data/anonymized_full_release_competition_dataset/problem2skill', 'w', encoding='utf-8') as f:
        f.write(str(problem2skill))


def parse_student_seq(student, skill2id, problem2id, it2id, at2id):
    """
    解析单个学生的序列数据，包括技能、答题结果、问题 ID、间隔时间和答题时间。

    参数:
        student (pd.DataFrame): 单个学生的数据。
        skill2id (dict): 技能 ID 映射表。
        problem2id (dict): 问题 ID 映射表。
        it2id (dict): 间隔时间 ID 映射表。
        at2id (dict): 答题时间 ID 映射表。

    返回值:
        tuple: 包含技能序列、答题结果序列、问题 ID 序列、间隔时间序列和答题时间序列的元组。
    """
    seq = student
    s = [skill2id[q] for q in seq.skill.tolist()]
    a = seq.correct.tolist()
    p = [problem2id[p] for p in seq.problemId.tolist()]

    it = [0]
    startTime = np.array(seq.startTime)
    for i in range(1, len(startTime)):
        item = (startTime[i] - startTime[i - 1]) // 60
        if item > 43200:
            item = 43200
        it.append(it2id[item])

    at = [at2id[int(x)] for x in seq.timeTaken.tolist()]
    return s, a, p, it, at


def parse_all_seq(students, data, skill2id, problem2id, it2id, at2id):
    """
    解析所有学生的序列数据。

    参数:
        students (list): 学生 ID 列表。
        data (pd.DataFrame): 数据集。
        skill2id (dict): 技能 ID 映射表。
        problem2id (dict): 问题 ID 映射表。
        it2id (dict): 间隔时间 ID 映射表。
        at2id (dict): 答题时间 ID 映射表。

    返回值:
        list: 包含所有学生序列数据的列表。
    """
    all_sequences = []
    for student_id in tqdm.tqdm(students, '解析学生序列:\t'):
        student_sequence = parse_student_seq(data[data.studentId == student_id], skill2id, problem2id, it2id, at2id)
        all_sequences.append(student_sequence)
    return all_sequences


def split_train_test_data(sequences):
    """
    将序列数据划分为训练集和测试集。

    参数:
        sequences (list): 所有学生的序列数据。

    返回值:
        tuple: 包含训练集和测试集的元组。
    """
    train_data, test_data = train_test_split(sequences, test_size=0.2, random_state=10)
    # return np.array(train_data), np.array(test_data)
    return train_data, test_data


def write_sequences_to_file(sequences, trg_path):
    """
    将序列数据写入文件。

    参数:
        sequences (list): 序列数据。
        trg_path (str): 目标文件路径。
    """
    with open(trg_path, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, '写入数据到文件: %s' % trg_path):
            s_seq, a_seq, p_seq, it_seq, at_seq = seq
            seq_len = len(s_seq)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(s) for s in s_seq]) + '\n')
            f.write(','.join([str(a) for a in a_seq]) + '\n')
            f.write(','.join([str(p) for p in p_seq]) + '\n')
            f.write(','.join([str(i) for i in it_seq]) + '\n')
            f.write(','.join([str(a) for a in at_seq]) + '\n')


def main():
    """
    主函数：执行数据预处理、序列解析、数据分割和文件写入等操作。
    """
    # 下载数据集（如果需要）
    download_dataset_if_needed()

    # 加载并预处理数据
    data = load_and_preprocess_data()

    # 生成 ID 映射表
    skill2id, problem2id, at2id = generate_id_mappings(data)

    # 计算间隔时间
    it2id = calculate_interval_time(data)

    # 建立问题到技能的映射关系
    map_problem_to_skill(data, skill2id, problem2id)

    # 解析所有学生的序列数据
    students = data.studentId.unique()
    sequences = parse_all_seq(students, data, skill2id, problem2id, it2id, at2id)

    # 划分训练集和测试集
    train_data, test_data = split_train_test_data(sequences)

    # 将数据写入文件
    kfold = KFold(n_splits=5, shuffle=True, random_state=10)
    idx = 0
    for train_data_1, valid_data in kfold.split(train_data):
        # write_sequences_to_file(train_data[train_data_1],
        #                         '../../data/anonymized_full_release_competition_dataset/train' + str(idx) + '.txt')
        # write_sequences_to_file(train_data[valid_data],
        #                         '../../data/anonymized_full_release_competition_dataset/valid' + str(idx) + '.txt')
        # idx += 1

        write_sequences_to_file([train_data[i] for i in train_data_1],
                                '../../data/anonymized_full_release_competition_dataset/train' + str(idx) + '.txt')
        write_sequences_to_file([train_data[i] for i in valid_data],
                                '../../data/anonymized_full_release_competition_dataset/valid' + str(idx) + '.txt')
        idx += 1

    write_sequences_to_file(test_data, '../../data/anonymized_full_release_competition_dataset/test.txt')


if __name__ == "__main__":
    main()
