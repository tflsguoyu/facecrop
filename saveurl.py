import requests
import cv2
import os

foldername = 'downloads/'

for i in range(2500,3001):
    for j in range(1,999):
        url = 'http://www.wxx0702.cn/Datas/%d/%d.jpg' % (i,j)
        filename = url.split('/')[-2] + '_' + url.split('/')[-1]
        fn = foldername + filename

        img_data = requests.get(url).content
        with open(fn, 'wb') as handler:
            handler.write(img_data)
        img = cv2.imread(fn)
        try:
            img.shape
            print('download ' + url)
        except:
            os.remove(fn)
            print(url + ' can not be read')
            break 
