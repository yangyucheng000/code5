# coding:utf-8

import pandas as pd


def main():

    df = pd.read_excel('test.xlsx', sheet_name='Sheet1', header=None)
    tag = []
    tail = []
    for i in range(24):
        a = df.values[:, i]
        tag.append(sum(a))
    for i in range(10):
        tail[i] = tag.sort()[i]

    res = []
    for i in range(2000):
        item = df.values[i, :]
        temp = []
        for inx, j in enumerate(item):
            if j == 1 and inx in tail:
                temp.append((tag[inx], inx))
                break
            elif j == 24 and inx not in tail:
                temp.append((tag[inx],inx))
            temp.sort()
            # max or min
            temp.sort(reverse=True )
            # temp.sort()
        print(temp)
        res.append(temp[0][1])
    print(res)
    with open('test.txt', 'w', encoding='utf-8') as fp:
        fp.write(res)


if __name__ == '__main__':
    main()





