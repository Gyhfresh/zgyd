import cv2

import os,json
from snownlp import SnowNLP



path="./faxueyuan"  #待读取的文件夹
path_list=os.listdir(path)
path_list.sort() #对读取的路径进行排序
for filename in path_list:
    num = 0
    i=0
    percent = 0
    f = open('./faxueyuan/'+filename, 'r', encoding='utf8')
    for line in f.readlines():
        i+=1
        data = json.loads(line)
        text = data['text']
        # print(text)
        s=SnowNLP(text)
        score=s.sentiments
        num+=score
        name=data['theme']
        if float(score)<0.7:
            percent+=1
        x=percent/i
        y=57.99*x*x+47.68*x-1.75
        if y >100:
            y=100

    print(name,num/i,x,y)

    # y = 57.99x2 + 47.68x - 1.7526