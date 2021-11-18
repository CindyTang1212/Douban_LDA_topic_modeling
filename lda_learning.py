# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 4:29 PM
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


# 预处理
dicts_path = 'dicts/dictionaries'
stop_words_path = "dicts/stopwords-master"
# 加载用户字典
my_words = ['九九六', '公务员', '内卷', '2020', '955', '找工作', '考编', '三十五岁', '宇宙尽头', '找不到工作',
            '秋招', '考公', '体制内', '稳定', '躺平', '985废物',
            '小镇做题家', '福利待遇', '同辈压力',
            '一线城市', '二线城市', '十八线城市', '找工作', '应届生', '组员朋友', '事业单位']
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


# 分词处理
def my_cut(text):
    result = ''
    try:
        for w in jieba.cut(text):
            if w not in stop_words:
                result += ' ' + w
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
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    docres = lda.fit_transform(tf)
    print('模型训练完成！')
    return tf, tf_vectorizer, lda, docres


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
    # 分词处理
    df = pd.read_csv('post_content/all.csv', error_bad_lines=False)
    df['content_cutted'] = df['页面标题'].apply(my_cut) + df['文本'].apply(my_cut) + df['高赞评论1'].apply(my_cut) + df[
        '高赞评论2'].apply(my_cut)
    content = df['content_cutted']
    content.to_csv('content_cutted.csv')
    print('切割完成！')

    # 建立lda模型
    n_topics = 5
    df = pd.read_csv('content_cutted.csv', error_bad_lines=False)
    tf, tf_vectorizer, lda, docres = lda_learning(n_features=1500, n_topics=n_topics,
                                                  contents=df['content_cutted'].values.astype('U'))
    tf_feature_names = tf_vectorizer.get_feature_names()

    # 输出结果
    print_top_words(lda, tf_feature_names, n_top_words=20)

    # 获取可视化结果
    pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.save_html(pic, 'result/' + 'lda_pass' + str(n_topics) + '.html')
