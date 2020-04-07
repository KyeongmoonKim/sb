import cv2
import os

aug_dir = '../results/VOC2007/Main/'
img_dir = './visual/'
fd = open('./ImageSets/Main/test.txt', 'r')
name_list = []

while True:
  line = fd.readline()
  if not line: break
  line = line[:len(line)-1]
  img = cv2.imread('./JPEGImages/'+line+'.jpg')
  cv2.imwrite(img_dir+line+'.jpg', img)
  
  
  
  
fd.close() #move part

dir = os.path.abspath(aug_dir)
for (root, dirs, files) in os.walk(dir):
    for file in files:
        name_list.append(aug_dir+file)
        print(file)
        splited = file.split('_')
        file_name = splited[4]
        fd = open(aug_dir+file, 'r')
        label = file_name[:len(file_name)-4]
        print("label "+label+" tag start")
        curr = None
        img = None
        while True:
            line = fd.readline()
            if not line:
              cv2.imwrite(img_dir+curr+'.jpg', img)
              break
            line = line[:len(line)-1]
            temp = line.split(' ')
            #img update
            if curr is None: #f
                curr = temp[0]
                img = cv2.imread(img_dir+curr+'.jpg')
            elif curr != temp[0]:
                cv2.imwrite(img_dir+curr+'.jpg', img)
                curr = temp[0]
                img = cv2.imread(img_dir+curr+'.jpg')
            score = float(temp[1])
            if score >= 0.5:
                cv2.rectangle(img, (int(temp[2].split('.')[0]), int(temp[3].split('.')[0])), (int(temp[4].split('.')[0]), int(temp[5].split('.')[0])), (0, 255, 0), 3)
                cv2.putText(img, label+' '+temp[1], (int(temp[2].split('.')[0]), int(temp[3].split('.')[0])), fontFace=2, fontScale=0.5, color=(0, 0, 0)) 
        print("label "+label+" tag finish")
        fd.close()
num = len(name_list)
for i in range(0, num):
  os.remove(name_list[i])
