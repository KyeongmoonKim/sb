import os
import xml.etree.ElementTree as ET
import cv2

in_dir = './data/imgs/'
img_dir = './data/VOCdevkit2007/VOC2007/JPEGImages/'
anno_dir = './data/VOCdevkit2007/VOC2007/Annotations/'
fd = open('./data/VOCdevkit2007/VOC2007/ImageSets/Main/', 'wt')

count = 0
dir_temp = os.path.abspath(in_dir)
for (root, dirs, files) in os.walk(dir_temp):
    for file in files:
        ext = file[len(file)-3:len(file)]
        if ext == 'jpg':
            img = cv2.imread(in_dir+file)
            cv2.imwrite(img_dir+'in'+str(count)+'.jpg', img)
            count = count+1
try:
    os.remove('./data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl')
except:
    print("No such file!")

dir = os.path.abspath(img_dir)
tree = ET.parse(anno_dir+'dummy.xml')
for (root, dirs, files) in os.walk(dir):
    for file in files:
        if(file[:2]=='in'):
            name = file[:len(file)-4]
            fd.write(name+'\n')
            tree.write(anno_dir+name+'.xml')

fd.close()