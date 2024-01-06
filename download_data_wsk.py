import pandas
import requests
import time
import os

def crawl_data(url):
    try:
        #response = requests.get(url)
        img = requests.get(url, timeout=(5,7))
        # 在这里处理网页数据的逻辑
        return img
    except requests.exceptions.Timeout:
        print("连接超时，等待一段时间后重新连接...")
        time.sleep(5)  # 等待5秒后重新连接
        return crawl_data(url)
    except requests.exceptions.RequestException as e:
        print("连接失败:", str(e))
        #return None
        return 100


book = pandas.read_csv('./file/test_data_8900_with_plan_whole.csv', encoding="gb2312")
img_root = "F://MyProject//乐信合作//各类数据//202307pbn时长数据//suoluetu"

thumbnails = book['region图']
indexes = book['id']
colored_images = book['完成图']

n = 0
for i in range(0, len(indexes)):
    index = indexes[i]

    img_name = os.path.join(img_root, index + ".png")
    thumbnail_url = thumbnails[i]
    if str(thumbnail_url) != 'nan':
        if os.path.exists(img_name):
            print(f'{i + 1}/{len(indexes)}, {index+".png"} already exists...')
            continue

        thumbnail_img = crawl_data(thumbnail_url)
        with open(img_name, 'wb') as fp:
            fp.write(thumbnail_img.content)

        n = n+1

    print('已下载',n, '副图片')