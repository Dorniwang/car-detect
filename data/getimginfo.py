import os

pos_path = "./pos"
neg_path = "./neg"

pos_img_path = []
neg_img_path = []

pathDirPos = os.listdir(pos_path)
pathDirNeg = os.listdir(neg_path)


file_pos_path = "./pos_img_info.txt"
file_neg_path = "./neg_img_info.txt"

for item in pathDirNeg:
    neg_img_path.append(item)
    
for item in pathDirPos:
    pos_img_path.append(item)
    

#print pos_img_path
with open(file_pos_path,'w') as f:
    for item in pos_img_path:
        f.write('%s%s' % (item, os.linesep))
f.close()
with open(file_neg_path,'w') as f1:
    for item in neg_img_path:
        f1.write('%s%s' % (item, os.linesep))
f1.close()


