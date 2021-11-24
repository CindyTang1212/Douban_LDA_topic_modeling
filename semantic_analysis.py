# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 1:09 PM
# @Author  : Cindy Tang
# @FileName: semantic_analysis.py
# @Software: PyCharm
# @Email    ：txdgz0624@gmail.com
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
import matplotlib.pyplot as plt
import jieba.posseg as psg
import re
import openpyxl

# 预处理
dicts_path = 'dicts/dictionaries'
stop_words_path = "dicts/stopwords-master"
# 加载用户字典
my_words = ['九九六', '公务员', '内卷', '2020', '955', '找工作', '考编', '三十五岁', '宇宙尽头', '找不到工作', '省考',
            '秋招', '考公', '体制内', '稳定', '躺平',
            '985废物', '小镇做题家', '福利待遇', '同辈压力', '国考',
            '一线城市', '二线城市', '十八线城市', '找工作', '应届生', '组员朋友', '事业单位', '性别歧视']
for i in my_words:
    jieba.add_word(i)
dicts_files = os.listdir(dicts_path)
for file in dicts_files:
    jieba.load_userdict(dicts_path + '/' + file)
print('字典添加完成！')
# 加载停用词
stop_words = []
stop_words_files = os.listdir(stop_words_path)
for file in stop_words_files:
    with open(stop_words_path + '/' + file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
print('停用词添加完成！')


def cut_long_text(text):
    result = []
    try:
        for t in text.split('。'):
            result.append(my_cut(t))
    except:
        result.append('')
    return result


def my_cut(text):
    flags = ['a', 'n', 'vn']
    raw_result = ''
    result = ''
    try:
        for w in jieba.cut(text):
            if w in my_words:
                result += ' ' + w
            elif w not in stop_words:
                raw_result += ' ' + re.sub(u'[^\u4e00-\u9fa5]', '', w)
        for w in psg.cut(raw_result):
            if w.flag in flags and len(w.word) > 1:
                result += ' ' + w.word
    except:
        result = ''
    return result


def str2csv(filePath, s, x):
    """
    将字符串写入到本地csv文件中
    :param filePath: csv文件路径
    :param s: 待写入字符串(逗号分隔格式)
    """
    if x == 'node':
        with open(filePath, 'w', encoding='utf-8') as f:
            f.write("label,weight\r")
            f.write(s)
        print('写入文件成功,请在' + filePath + '中查看')
    else:
        with open(filePath, 'w', encoding='gbk') as f:
            f.write("source,target,weight\r")
            f.write(s)
        print('写入文件成功,请在' + filePath + '中查看')


def sortDictValue(dict, is_reverse):
    """
    将字典按照value排序
    :param dict: 待排序的字典
    :param is_reverse: 是否按照倒序排序
    :return s: 符合csv逗号分隔格式的字符串
    """
    # 对字典的值进行倒序排序,items()将字典的每个键值对转化为一个元组,key输入的是函数,item[1]表示元组的第二个元素,reverse为真表示倒序
    tups = sorted(dict.items(), key=lambda item: item[1], reverse=is_reverse)
    s = ''
    for tup in tups:  # 合并成csv需要的逗号分隔格式
        s = s + tup[0] + ',' + str(tup[1]) + '\n'
    return s


def build_matrix(co_authors_list, is_reverse):
    """
    根据共同列表,构建共现矩阵(存储到字典中),并将该字典按照权值排序
    :param co_authors_list: 共同列表
    :param is_reverse: 排序是否倒序
    :return node_str: 三元组形式的节点字符串(且符合csv逗号分隔格式)
    :return edge_str: 三元组形式的边字符串(且符合csv逗号分隔格式)
    """
    node_dict = {}  # 节点字典,包含节点名+节点权值(频数)
    edge_dict = {}  # 边字典,包含起点+目标点+边权值(频数)
    count = 1
    # 第1层循环,遍历整表的每行信息
    for row_authors in co_authors_list:
        row_authors_list = row_authors.split(' ')  # 依据','分割每行,存储到列表中
        # 第2层循环
        for index, pre_au in enumerate(row_authors_list):  # 使用enumerate()以获取遍历次数index
            # 统计单个词出现的频次
            if pre_au not in node_dict:
                node_dict[pre_au] = 1
            else:
                node_dict[pre_au] += 1
            # 若遍历到倒数第一个元素,则无需记录关系,结束循环即可
            if pre_au == row_authors_list[-1]:
                break
            connect_list = row_authors_list[index + 1:]
            # 第3层循环,遍历当前行词后面所有的词,以统计两两词出现的频次
            for next_au in connect_list:
                A, B = pre_au, next_au
                # 固定两两词的顺序
                # 仅计算上半个矩阵
                if A == B:
                    continue
                if A > B:
                    A, B = B, A
                key = A + ',' + B  # 格式化为逗号分隔A,B形式,作为字典的键
                # 若该关系不在字典中,则初始化为1,表示词间的共同出现次数
                if key not in edge_dict:
                    edge_dict[key] = 1
                else:
                    edge_dict[key] += 1
        count += 1
    # 对得到的字典按照value进行排序
    node_str = sortDictValue(node_dict, is_reverse)  # 节点
    print(node_str)
    edge_str = sortDictValue(edge_dict, is_reverse)  # 边
    print(edge_str)
    return node_str, edge_str


def get_co_occurance_results(edge_file, node_file):
    edge_str = pd.read_csv(edge_file, encoding='gbk')
    edge_str1 = edge_str[edge_str['weight'] > 70]
    Source = edge_str1['source'].tolist()
    Target = edge_str1['target'].tolist()
    co = Source + Target
    co = list(set(co))
    node_str = pd.read_csv(node_file, encoding='utf-8')
    # node_str
    node_str = node_str[node_str['label'].isin(co)]
    node_str['id'] = node_str['label']
    node_str = node_str[['id', 'label', 'weight']]  # 调整列顺序
    # node_str
    node_str.to_csv(path_or_buf='result/semantic/node' + topic + '_Final.csv', index=False)  # 写入csv文件
    edge_str1.to_csv(path_or_buf='result/semantic/edge' + topic + '_Final.csv', index=False)  # 写入csv文件


if __name__ == '__main__':
    topic = 'Topic1'
    filePath1 = 'result/semantic/node' + topic + '.csv'
    filePath2 = 'result/semantic/edge' + topic + '.csv'
    # 读取csv文件获取数据并存储到列表中
    df = pd.read_csv('result/lda_result.csv')
    df_ = df[df['Topic_Category'] == topic]
    result = df_['文本'].apply(cut_long_text) + df_['高赞评论1'].apply(cut_long_text) + df_['高赞评论2'].apply(cut_long_text)
    word_lists = []
    for r in result:
        for s in r:
            word_lists.append(s)
    word_lists = [w.strip() for w in word_lists if len(w) > 4]
    # 根据共同词列表, 构建共现矩阵(存储到字典中), 并将该字典按照权值排序
    node_str, edge_str = build_matrix(word_lists, is_reverse=True)
    # 将字符串写入到本地csv文件中
    str2csv(filePath1, node_str, 'node')
    str2csv(filePath2, edge_str, 'edge')
    get_co_occurance_results(node_file=filePath1, edge_file=filePath2)