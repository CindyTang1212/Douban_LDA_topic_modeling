# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 4:29 PM
# @Author  : Cindy Tang
# @FileName: lda_learning.py
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


# # 预处理
# dicts_path = 'dicts/dictionaries'
# stop_words_path = "dicts/stopwords-master"
# # 加载用户字典
# my_words = ['九九六', '公务员', '内卷', '2020', '955', '找工作', '考编', '三十五岁', '宇宙尽头', '找不到工作', '省考'
#             '秋招', '考公', '体制内', '稳定', '躺平', '985废物', '小镇做题家', '福利待遇', '同辈压力', '国考',
#             '一线城市', '二线城市', '十八线城市', '找工作', '应届生', '组员朋友', '事业单位', '性别歧视']
# for i in my_words:
#     jieba.add_word(i)
# dicts_files = os.listdir(dicts_path)
# for file in dicts_files:
#     jieba.load_userdict(dicts_path + '/' + file)
# print('字典添加完成！')
# # 加载停用词
# stop_words = []
# stop_words_files = os.listdir(stop_words_path)
# for file in stop_words_files:
#     with open(stop_words_path + '/' + file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             stop_words.append(line.strip())
# print('停用词添加完成！')


# 分词处理
def my_cut(text):
    flags = ['v', 'a', 'nz', 'n', 'nt', 'vn', 'nr', 'ns']
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


def lda_learning(n_features, n_topics, contents):
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10)
    tf = tf_vectorizer.fit_transform(contents)
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    docres = pd.DataFrame(lda.fit_transform(tf))
    print('模型训练完成！')
    return tf, tf_vectorizer, lda, docres


def get_perplexity(n_features, n_max_topics, contents):
    # 建立lda模型，计算模型困惑度
    plexs = []
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10)
    tf = tf_vectorizer.fit_transform(contents)
    for i in range(1, n_max_topics + 1):
        print(i, end=' : ')
        lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                        learning_method='online',
                                        learning_offset=50, random_state=0)
        lda.fit(tf)
        print(lda.perplexity(tf), end='\n')
        plexs.append(lda.perplexity(tf))
    n_t = n_max_topics  # 区间最右侧的值。注意：不能大于n_max_topics
    x = list(range(1, n_max_topics + 1))
    plt.plot(x, plexs[0:])
    plt.xlabel("number of topics")
    plt.ylabel("perplexity")
    plt.savefig('result/perplexity.jpg')
    plt.show()


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def doc_category(docres):
    # 文档所属每个类别的概率
    LDA_corpus = np.array(docres)
    # 对比所属两个概率的大小，确定属于的类别
    result1 = np.argmax(LDA_corpus, axis=1).tolist()  # 返回沿轴axis最大值的索引，axis=1代表行；最大索引即表示最可能表示的数字是多少
    result2 = np.max(LDA_corpus, axis=1).tolist()


if __name__ == '__main__':
    # # 分词处理
    # df = pd.read_csv('post_content/all.csv', error_bad_lines=False)
    # df['content_cutted'] = df['页面标题'].apply(my_cut) + df['文本'].apply(my_cut) + df['高赞评论1'].apply(
    #     my_cut) + df['高赞评论2'].apply(my_cut)
    # df.to_csv('result/content_cutted.csv')
    # print('切割完成！')

    # # 困惑度计算
    # df = pd.read_csv('content_cutted.csv', error_bad_lines=False).drop_duplicates(subset=['content_cutted'])
    # get_perplexity(n_max_topics=10, contents=df['content_cutted'].values.astype('U'))

    # 建立主题模型
    n_topics = 4
    df = pd.read_csv('result/content_cutted.csv', error_bad_lines=False)
    tf, tf_vectorizer, lda, docres = lda_learning(n_features=500, n_topics=n_topics,
                                                  contents=df['content_cutted'].values.astype('U'))
    tf_feature_names = tf_vectorizer.get_feature_names()

    # 输出结果
    print_top_words(lda, tf_feature_names, n_top_words=20)

    # 获取可视化结果
    pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.save_html(pic, 'result/' + 'lda_pass' + str(n_topics) + '.html')

    # 将结果存储入excel
    cols = []
    for count in range(n_topics):
        cols.append('Topic' + str(count + 1))
    docres.columns = cols
    df['Topic_Category'] = docres.idxmax(axis=1)
    df['Topic_Weight'] = docres.max(axis=1)
    df.to_csv('result/lda_result.csv', columns=['页面标题', '文本', '高赞评论1', '高赞评论2', 'Topic_Category', 'Topic_Weight'])
    writer = pd.ExcelWriter('result/weight.xlsx')  # 关键2，创建名称为hhh的excel表格
    df.to_excel(writer, 'page_1',
                float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4
