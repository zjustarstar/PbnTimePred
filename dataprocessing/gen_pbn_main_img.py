# 生成实际填色时的图像

import json
from PIL import Image
import time
import pandas
import random
import cv2
import multiprocessing
import numpy as np
import os


def create_main_thread(args):
    i, total, img_path_root, save_path, img_name, block_index, block_color = args
    save_img_name = os.path.join(save_path, img_name[i])

    print(f'process {i + 1}/{total}')

    # 如果文件已存在，直接跳过
    if os.path.exists(save_img_name):
        return

    # 处理图片
    img_path = os.path.join(img_path_root, img_name[i])
    if not os.path.exists(img_path):
        print(f'{img_path} do not exist!')
        return

    blk_index = block_index[i]
    blk_color = block_color[i]
    im = Image.open(img_path)
    if im.mode == 'RGBA':
        im = np.array(im)
        rgb_info = im[:, :, :3]
        a_info = im[:, :, 3]
        rgb_info[a_info == 0] = [255, 255, 255]
        im = Image.fromarray(rgb_info)

    # 根据plan组的数据进行填色
    im_array = np.array(im)
    new_img = np.zeros_like(im_array)

    for ind, blocks in enumerate(blk_index):
        blk = blocks.split(',')
        rgb_list = []
        for b in blk:
            bb = int(b) % 256
            gg = int(b) // 256
            rr = 0
            mask = np.all(im_array == (rr, gg, bb), axis=-1)

            clr = blk_color[ind]
            new_img[:, :, 0][mask] = clr[0]
            new_img[:, :, 1][mask] = clr[1]
            new_img[:, :, 2][mask] = clr[2]

            # rgb_list.append([rr, gg, bb])

        # pixels = np.array(rgb_list)
        # mask = np.zeros(im_array.shape[:2], dtype=bool)
        # for pv in pixels:
        #     mask = np.any([mask, np.all(im_array == pv, axis=-1)], axis=0)
        #
        # clr = blk_color[ind]
        # new_img[:, :, 0][mask] = clr[0]
        # new_img[:, :, 1][mask] = clr[1]
        # new_img[:, :, 2][mask] = clr[2]

    s = Image.fromarray(new_img.astype(np.uint8))
    save_img_name = os.path.join(save_path, img_name[i])
    s.save(save_img_name)



def create_main_thread2(args):
    i, total, img_path_root, save_path, img_name, block_index, block_color = args
    print(f'process {i + 1}/{total}')

    print(i)
    save_img_name = os.path.join(save_path, img_name[i])

    # 如果文件已存在，直接跳过
    if os.path.exists(save_img_name):
        return

    # 处理图片
    img_path = os.path.join(img_path_root, img_name[i])
    if not os.path.exists(img_path):
        print(f'{img_path} do not exist!')
        return

    blk_index = block_index[i]
    blk_color = block_color[i]
    im = Image.open(img_path)
    if im.mode == 'RGBA':
        im = np.array(im)
        rgb_info = im[:, :, :3]
        a_info = im[:, :, 3]
        rgb_info[a_info == 0] = [255, 255, 255]
        im = Image.fromarray(rgb_info)

    # 根据plan组的数据进行填色
    im_array = np.array(im)
    new_img = np.copy(im_array)

    for ind, clr in enumerate(blk_color):
        blk = blk_index[ind].split(',')
        rgb_list = []
        for b in blk:
            bb = int(b) % 256
            gg = int(b) // 256
            rr = 0
            # mask = np.all(im_array == (rr, gg, bb), axis=-1)
            #
            # clr = blk_color[ind]
            # new_img[:, :, 0][mask] = clr[0]
            # new_img[:, :, 1][mask] = clr[1]
            # new_img[:, :, 2][mask] = clr[2]

            rgb_list.append([rr, gg, bb])

        pixels = np.array(rgb_list)
        mask = np.zeros(im_array.shape[:2], dtype=bool)
        for pv in pixels:
            mask = np.any([mask, np.all(im_array == pv, axis=-1)], axis=0)

        clr = blk_color[ind]
        new_img[:, :, 0][mask] = clr[0]
        new_img[:, :, 1][mask] = clr[1]
        new_img[:, :, 2][mask] = clr[2]

    s = Image.fromarray(new_img.astype(np.uint8))
    save_img_name = os.path.join(save_path, img_name[i])
    s.save(save_img_name)


def create_pbnimg_by_data(data, img_path_root, save_path, num=-1):
    '''
    :param data:从json文件中读取的数据，等同于excel中一条记录
    :param num: 每次处理几张。-1表示所有
    :return:
    '''
    img_name = []
    block_index = []
    block_color = []
    time = []
    for key, value in data.items():
        img_name.append(key)
        plan = value[2]
        block_index.append(plan['blocks_index'])
        block_color.append(plan['blocks_color'])
        time.append(value[3])

    total = len(data)
    if num != -1:
        total = num

        # 创建一个进程池
    pool = multiprocessing.Pool(2)
    # 创建一个Manager对象
    manager = multiprocessing.Manager()
    # 创建一个可在多个进程之间共享的锁对象
    lock = manager.Lock()
    # 要处理的数据
    data1 = list(range(0, total))

    # 使用map进行并行处理
    pool.map(create_main_thread, [(item, total, img_path_root, save_path, img_name, block_index, block_color) for item in data1])

    # 关闭进程池
    pool.close()
    pool.join()
    # for i in range(total):
    #     print(f'process {i + 1}/{total}')
    #     create_main_thread2(i, img_path_root, save_path, img_name, block_index, block_color)


if __name__ == '__main__':
    # cvs_file_path = "F://MyProject//乐信合作//各类数据//202307pbn时长数据//test_data_8900_with_plan_whole.csv"
    json_file_path = "..//file//test.json"

    data = {}
    # -------------------- 将所有数据写入whole_original.json文件 ---------------------------------
    with open(json_file_path, "r") as r:
        data = json.load(r)

    img_root = "F://MyProject//乐信合作//各类数据//202307pbn时长数据//suoluetu"
    save_root = "./"

    start_time = time.time()
    create_pbnimg_by_data(data, img_root, save_root)
    end_time = time.time()
    time1 = round(end_time - start_time)
    print(f'{time1//60} min, {time1%60} sec required')
    print("done")


