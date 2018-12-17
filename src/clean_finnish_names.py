import os
import sys

root_path = os.path.dirname(os.path.abspath(os.getcwd()))
raw_path = os.path.join(root_path,'raw')
data_path = os.path.join(root_path,'data') # to store data processed

names = []
with open(os.path.join(raw_path,'finnish_names.txt'),'r') as f:
    for line in f.readlines():
        a_name = str(line.rstrip())
        a_name = a_name.lower()
        a_name = a_name[0].upper()+a_name[1:]
        names.append(a_name)
names = sorted(list(set(names)))

with open(os.path.join(data_path,'finnish_names.txt'),'w') as f:
    f.write("\n".join(names))