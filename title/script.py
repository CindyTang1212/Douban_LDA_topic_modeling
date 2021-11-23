# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 11:29 PM
# @Author  : Cindy Tang
# @FileName: script.py
# @Software: PyCharm
# @Email    ：txdgz0624@gmail.com
import pandas as pd

df = pd.read_csv('所有标题.csv')
df.to_excel('所有标题.xlsx')