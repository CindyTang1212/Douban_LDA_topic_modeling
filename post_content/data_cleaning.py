import csv
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('1.csv', error_bad_lines=False)
    df.drop_duplicates()
    for index, row in df.iterrows():
        print(index)
    # rows = [x for x, col in enumerate(df['文本']) if len(df.iat[x, col] < 10)]
    # print('经过清洗后，数据行数变成' + len(rows))