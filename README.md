# car-detect
car detect using opencv in c++

Download the data set from  https://cogcomp.cs.illinois.edu/Data/Car/ , and build the dir according to the instruct in data/README.md.

Then run getimginfo.py to create two files pos_img_info.txt and neg_img_info.txt.

After preprocess on dataset, we follow the following steps to build:
  1. cd build
  2. cmake ..
  3. make
All config have already set in CMakeLists.txt, you may want to change the opencv lib dir.

First run train-classifier to train the model, then run test-classifier to test a new picture.

Opencv has the ml modular itself, we use the svm from ml modular. But we need the confidence score to perform nms, and svm in opencv is not provide this interface, so I encapsulate the svm.cpp file and add the 'decision_func_score' function to give the confidence score.
