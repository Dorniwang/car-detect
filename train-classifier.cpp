#include <iostream>
#include "opencv/cv.h"
#include "opencv/ml.h"
#include "opencv/highgui.h"
#include "opencv2/objdetect/objdetect.hpp"
#include <string>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    string path_pos_feat = "../data/feat/pos";
    string path_neg_feat = "../data/feat/neg";
    
    //正负样本图片名称列表
    ifstream pos_img("../data/pos_img_info.txt");
    ifstream neg_img("../data/neg_img_info.txt");
    if(pos_img == NULL)
    {
        cout<<" open pos img info filed !"<<endl;
    }
    if(neg_img == NULL)
    {
        cout<<" open neg img info filed !"<<endl;
    }
    //正负样本图片数量
    int numofPosImg = 0;
    int numofNegImg = 0;
    
    string tempfilename;
    
    //正负样本图片路径
    vector<string> posImg_path;
    vector<string> negImg_path;
    
    while(pos_img)
    { 
        if(getline(pos_img, tempfilename))
        {
            numofPosImg++;
            string path = "../data/pos/" + tempfilename;
            posImg_path.push_back(path);
        }
    }
    pos_img.close();
    while(neg_img)
    {
        if(getline(neg_img, tempfilename))
        {
            numofNegImg++;
            string path = "../data/neg/" + tempfilename;
            negImg_path.push_back(path);
        }
    }
    neg_img.close();
    
    int numImg = numofNegImg + numofPosImg;
    CvMat *train_data, *target;
    
    //维度为1764,计算方式：64×64的图片，8×8像素的cell，每个cell9个bins，2×2个cell为一个
    //block，因此每个块内有4×9=36个特征。block的移动步长为8个像素，对于64×64的图片来说水平
    //方向有8个cell，7个扫描窗口，垂直方向也是8个cell，7个扫描窗口。因此特征维度为36×7×7=1764.
    train_data = cvCreateMat(numImg, 1764, CV_32FC1);
    
    //cvSetZero(train_data);
    //cvSet(train_data,cvScalarAll(0),0);
    
    target = cvCreateMat(numImg, 1, CV_32FC1);
    //cvSetZero(target);
    
    IplImage *src; //cvResize的原图片
    IplImage *trainImg = cvCreateImage(cvSize(64,64),8,3); //cvResize的目标图片大小
    //cout<<numImg<<endl;
    //提取hog特征
    for(int i = 0; i < numImg; i++)
    {
        float label;
        if(i<numofPosImg)
        {
            //如果index小于numofPosImg，则对正样本图片进行操作
            src = cvLoadImage(posImg_path[i].c_str(),1);
            if(src == NULL)
            {
                cout<<"can not load the image: "<<posImg_path[i].c_str()<<endl;
                continue;
            }
            label = 1.0;
        }
        else
        {
            //否则对负样本图片进行操作
            src = cvLoadImage(negImg_path[i-numofPosImg].c_str(),1);
            if(src == NULL)
            {
                cout<<"can not load the image: "<<posImg_path[i].c_str()<<endl;
                continue;
            }
            label = -1.0;
        }
        
        cvResize(src,trainImg);
        //参数列表为： 检测窗口大小，块大小（像素表示），块步长（像素表示），cell大小（像素表示），nbins
        HOGDescriptor *hog = new HOGDescriptor(cvSize(64,64), cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
        
        vector<float> descriptors; //结果数组
        //computer参数列表：输入图片，存储descriptors特征结果的vector，检测窗口的步进，padding
        hog->compute(trainImg,descriptors,Size(1,1),Size(0,0));
        cout<<"Hog dims"<<descriptors.size()<<endl;
        int col = 0;
        for(vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
        {
            cvmSet(train_data, i, col, *iter);
            col++;
        }
        //cout<<label<<endl;
        cvmSet(target, i, 0, label);
        if(label == 1.0)
            cout<<" end processing "<<posImg_path[i].c_str()<<" "<<"1"<<endl;
        else if(label == -1.0)
            cout<<" end processing "<<negImg_path[i-numofPosImg].c_str()<<" "<<"0"<<endl;
        
    }
    cout<<"test5"<<endl;
    //训练svm分类器
    
    CvSVM svm;
    //参数列表为：类型（CV_TERMCRIT_ITER和CV_TERMCRIT_EPS二值之一或二者的组合），最大迭代次数，结果的精确性
    CvTermCriteria criteria;
    criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
    //参数列表为： svm类型;核函数;核函数参数自由度（多项式）;核函数gamma参数（多项式/RBF/SIGMOID）;核函数的coef0参数（多项式/SIGMOID）
    //svm优化问题的c参数（C_SVC/EPS_SVR/NU_SVR）;svm优化问题的nu参数（NU_SVC/ONE_CLASS/NU_SVR）;svm优化问题的sigma参数（EPS_SVR）
    //C_SVC问题中的可选权重，分配给特定的类;解决约束二次优化问题部分实例的svm迭代训练过程的终止条件
    CvSVMParams param;
    param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria); 
    
    //训练 
    svm.train(train_data, target, NULL, NULL, param);
    svm.save("../model/svm_model.xml");
    return 0;
}