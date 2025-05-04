#!/usr/bin/env python
# coding: utf-8

from EduData import get_data
import random
import pandas as pd
from tqdm import tqdm
import chardet


def download_dataset(dataset_name, target_path):
    """
    下载指定的数据集到目标路径。

    参数:
        dataset_name (str): 数据集名称。
        target_path (str): 目标存储路径。
    """
    get_data(dataset_name, target_path)

def detect_encoding(file_path):
    """
    使用 chardet 库检测文件的编码。

    本函数以二进制模式读取文件，并利用 `chardet` 库来确定文件的编码格式。
    它会将检测到的编码以字符串形式返回。

    参数
    ----------
    file_path : str
        需要检测编码的文件路径。该路径应为程序可访问的有效文件路径。

    返回值
    -------
    str
        检测到的文件编码，以字符串形式返回。如果无法可靠地确定编码，
        返回值可能为 None 或空字符串，具体取决于 `chardet` 库的行为。

    抛出异常
    ------
    FileNotFoundError
        如果提供的文件路径指向一个不存在的文件，则会抛出此异常。
    PermissionError
        如果程序没有足够的权限读取文件，则会抛出此异常。
    IOError
        如果在尝试读取文件时发生输入/输出错误，则会抛出此异常。

    注意
    -----
    本函数依赖第三方库 `chardet` 来执行编码检测。
    请确保在运行此函数的环境中已安装并可用该库。

    检测到的编码准确性取决于文件的内容和大小。
    对于非常小的文件或使用非标准编码的文件，检测结果可能不够可靠。

    警告
    --------
    本函数会以二进制模式将整个文件读入内存。
    对于非常大的文件，这可能会导致较高的内存消耗。
    在处理大文件时请谨慎使用。
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_and_preprocess_data(file_path):
    """
    加载并预处理数据集，移除无效行并选择特定列。

    参数:
        file_path (str): CSV文件路径。

    返回:
        pd.DataFrame: 预处理后的数据集。
    """
    # detected_encoding = detect_encoding(file_path)
    # print(f"Detected encoding: {detected_encoding}")
    # Detected encoding: Windows-1252

    data = pd.read_csv(
        file_path,
        usecols=['order_id', 'user_id', 'skill_id', 'problem_id', 'correct'],
        encoding='latin1'
    ).dropna(subset=['skill_id', 'problem_id'])
    return data


def map_unique_ids(raw_list):
    """
    将唯一的原始ID映射到连续整数ID。

    参数:
        raw_list (list): 原始ID列表。

    返回:
        dict: 映射字典，键为原始ID，值为新ID。
    """
    return {p: i + 1 for i, p in enumerate(raw_list)}


def print_dataset_statistics(num_skills, num_problems):
    """
    打印数据集统计信息，如技能和问题的数量。

    参数:
        num_skills (int): 技能数量。
        num_problems (int): 问题数量。
    """
    print(f"技能数量: {num_skills}")
    print(f"问题数量: {num_problems}")


def parse_student_sequence(student_data, skills_map, problems_map):
    """
    解析单个学生的序列数据，包括技能、问题和答案。

    参数:
        student_data (pd.DataFrame): 单个学生的行为数据。
        skills_map (dict): 技能ID映射。
        problems_map (dict): 问题ID映射。

    返回:
        tuple: 包含技能序列、问题序列和答案序列的元组。
    """
    seq = student_data.sort_values('order_id')
    skill_seq = [skills_map[q] for q in seq.skill_id.tolist()]
    problem_seq = [problems_map[q] for q in seq.problem_id.tolist()]
    answer_seq = seq.correct.tolist()
    return skill_seq, problem_seq, answer_seq


def parse_all_sequences(students, full_data, skills_map, problems_map):
    """
    解析所有学生的序列数据。

    参数:
        students (list): 学生ID列表。
        full_data (pd.DataFrame): 完整数据集。
        skills_map (dict): 技能ID映射。
        problems_map (dict): 问题ID映射。

    返回:
        list: 所有学生序列的列表。
    """
    all_sequences = []
    for student_id in tqdm(students, desc='解析学生序列'):
        student_data = full_data[full_data.user_id == student_id]
        student_sequence = parse_student_sequence(student_data, skills_map, problems_map)
        all_sequences.append(student_sequence)
    return all_sequences


def split_train_test(data, train_ratio=0.7, shuffle=True):
    """
    将数据集分割为训练集和测试集。

    参数:
        data (list): 待分割的数据。
        train_ratio (float): 训练集比例，默认为0.7。
        shuffle (bool): 是否打乱数据，默认为True。

    返回:
        tuple: 包含训练集和测试集的元组。
    """
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_ratio)
    return data[:boundary], data[boundary:]


def save_sequences_to_file(sequences, file_path):
    """
    将序列数据保存到文件中，格式为三行式（长度、问题、技能、答案）。

    参数:
        sequences (list): 序列数据。
        file_path (str): 目标文件路径。
    """
    with open(file_path, 'a', encoding='utf8') as f:
        for seq in tqdm(sequences, desc=f'写入文件 {file_path}'):
            skills, problems, answers = seq
            seq_len = len(skills)
            f.write(f"{seq_len}\n")
            f.write(','.join([str(q) for q in problems]) + '\n')
            f.write(','.join([str(q) for q in skills]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')


def main():
    # Step 1: 下载数据集
    # download_dataset("assistment-2009-2010-skill", "../../data")

    # Step 2: 加载并预处理数据
    file_path = '../../data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv'
    data = load_and_preprocess_data(file_path)

    # Step 3: 获取技能和问题ID的映射
    raw_skills = data.skill_id.unique().tolist()
    raw_problems = data.problem_id.unique().tolist()
    skills_map = map_unique_ids(raw_skills)
    problems_map = map_unique_ids(raw_problems)

    # Step 4: 打印数据集统计信息
    num_skills = len(raw_skills)
    num_problems = len(raw_problems)
    print_dataset_statistics(num_skills, num_problems)

    # Step 5: 解析所有学生的序列数据
    students = data.user_id.unique().tolist()
    sequences = parse_all_sequences(students, data, skills_map, problems_map)

    # Step 6: 分割训练集和测试集
    train_sequences, test_sequences = split_train_test(sequences)

    # Step 7: 保存序列数据到文件
    train_file = '../../data/2009_skill_builder_data_corrected/train_pid.txt'
    test_file = '../../data/2009_skill_builder_data_corrected/test_pid.txt'
    save_sequences_to_file(train_sequences, train_file)
    save_sequences_to_file(test_sequences, test_file)


if __name__ == "__main__":
    main()
