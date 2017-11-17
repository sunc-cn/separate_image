#!/usr/local/bin/python3.6
#encoding=utf-8

import os.path
import random
from smart_separate import *

def list_all_file(rootDir,all_file): 
    if not os.path.exists(rootDir) or not os.path.isdir(rootDir):
        return False
    try:
        for lists in os.listdir(rootDir): 
            path = os.path.join(rootDir, lists) 
            all_file.append(path)
            if os.path.isdir(path): 
                list_all_file(path,all_file) 
    except Exception as e:
        print(e)
        return False
    return True


def main(image_dir,dst_dir):
    all_files = []
    list_all_file(image_dir,all_files)
    for item in all_files:
        base_name =  os.path.basename(item)
        dst = smart_separate_ex(item)
        if dst != None:
            for sub in dst:
                (name,im) = sub
                new_path = dst_dir + "/" + name 
                im.save(new_path)
        pass

if __name__ == "__main__":
    image_dir = "./train_set"
    dst_dir = "./train_set_split"
    main(image_dir,dst_dir)
    pass

