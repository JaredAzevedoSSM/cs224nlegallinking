import json
import os 


def dir_files(directory: str):
    print(directory)
    dir_list = [x for x in os.listdir(path) if 'full' in x] 
    return dir_list


def data(dir_list: list):
    obs = []
    for file in dir_list:
        f = open(file, 'r')
        file_contents = f.read().splitlines()
        for elem in file_contents:
            obs.extend(json.loads(elem))
    return obs


def label(dictionary: dict):
    for i,sample in enumerate(dictionary):
        sample['label'] = 0
        if len(sample['matches']) != 0:
            sample['label'] = 1
        # print(i, sample, type(sample))
    return dictionary

if __name__ == "__main__":
    path = os.getcwd()
    dir_list = dir_files(path)
    # dir_list = dir_list[0:2]
    dictionary = data(dir_list)
    print(len(dictionary))
    # dictionary = dictionary[0:2]
    print(len(dictionary))
    dictionary = label(dictionary)
    print(dictionary[0:50])


    
    
