#include <iostream>
#include "opencv/cv.h"
#include "opencv/ml.h"
#include "opencv/highgui.h"
#include "opencv2/objdetect/objdetect.hpp"
#include <string>
#include <vector>
#include <fstream>
#include "nms.h"
#include "svm.h"

using namespace std;
using namespace cv;


void sliding_window(Mat image, vector<CvRect>& result, int* window_size, int* step_size)
{
    /*
        This function returns a patch of images of size equal
        to window_size.
        @ image: input image
        @ window_size: size of sliding window_size
        @ step_size: incremented size of window_size
        
        The function returns a CvRect, where
        @ x is the top-left x co-ordinate
        @ y is the top-left y co-ordinate
        @ width is the width of the sliding window rect
        @ height is the height of the sliding window rect
    */
    int sizeX = image.cols;
    int sizeY = image.rows;
    int win_width = window_size[0];
    int win_height = window_size[1];
    int incrementX = step_size[0];
    int incrementY = step_size[1];
    for(int y = 0; y + win_height < sizeY;)
    {
        for(int x = 0; x + win_width < sizeX;)
        {
            CvRect temp;
            temp.x = x;
            temp.y = y;
            temp.width = win_width;
            temp.height = win_height;
            x += incrementX;
            result.push_back(temp);
        }
        y += incrementY;
    }
}

int main(int argc, char** argv)
{
    int window_size[2] = {100,40};
    int step_size[2] = {10,10};
    int downscale = 1.25;
    IplImage* testImg = cvLoadImage("../data/testimg/test-0.pgm",1);
    //IplImage* testImg = cvLoadImage("pos-1.pgm", 1);
    DorniSVM svm;
    svm.load("../model/svm_model.xml");
    
    vector<detect> detections;
    int scale = 0;
    int t = 4;//塔顶图像的最小维数的对数值，即塔顶图像最小维数为16
    int n = testImg->width;//原图像宽度
    int m = testImg->height;//原图像高度
    //高斯金字塔的层数 为 log2（min（m，n）） - t
    int level = log((n>m)?m:n) / log(2) - t;
    Mat tmp, dest;
    tmp = testImg;
    dest = tmp;
    
    CvMat *testMat = cvCreateMat(1,1764,CV_32FC1);
    //获取svm的支持向量，进而计算分离超平面的法向量，每一个支持向量的维数
    //与hog特征维数相同，为1764，因此每个const float数组有1764个元素
    /*int support_vector_count = svm.get_support_vector_count();
    vector<const float*> support_vectors;
    for(int i = 0; i < support_vector_count; i++)
    {
        const float* temp = svm.get_support_vector(i);
        support_vectors.push_back(temp);
    }*/
    
    for(int i = 0; i < level; i++)
    {
        vector<detect> cd; // 临时存放当前尺度的detections
        
        /*
        pyrDown(tmp,dest,Size(tmp.cols/2,tmp.rows/2));
        
        if(dest.cols < window_size[0] && dest.rows < window_size[1])
            break;
        tmp.release();
        tmp = dest;
        IplImage* tmpImage;
        tmpImage = cvCreateImage(cvSize(dest.cols,dest.rows),8,3);
        tmpImage->imageData = (char*)dest.data;
        
        IplImage* dest1 = cvCreateImage(cvSize(64,64),8,3);
        cvResize(tmpImage, dest1);
        
        dest.release();
        dest = dest1;
        
        */
        
        vector<CvRect> result;
        sliding_window(dest,result,window_size,step_size);
        
        for(vector<CvRect>::iterator iter = result.begin(); iter != result.end(); iter++)
        {
            if(iter->width != window_size[0] && iter->height != window_size[1])
                continue;
            
            HOGDescriptor *hog = new HOGDescriptor(cvSize(64,64), cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
            
            vector<float> descriptors; //结果数组
            
            
            Rect rect(iter->x, iter->y, iter->width, iter->height);
            //cout<<rect<<" "<<dest.cols<<" "<<dest.rows<<endl;
            Mat image_roi = dest(rect);
            IplImage img_ = image_roi;
            IplImage* img_tmp = cvCloneImage(&img_);
            IplImage* img_roi = cvCreateImage(cvSize(64,64),8,3);
            cvResize(img_tmp, img_roi);
            Mat t = img_tmp;
            
            //imshow("test",t);
            //waitKey();  
            //compute参数列表：输入图片，存储descriptors特征结果的vector，检测窗口的步进，padding
            hog->compute(img_roi,descriptors,Size(1,1),Size(0,0));
            int col = 0;
            
            for(vector<float>::iterator iter1 = descriptors.begin(); iter1 != descriptors.end(); iter1++)
            {
                cvmSet(testMat, 0, col, *iter1);
                col++;
            }
            float pred = svm.predict(testMat);
            float confidence_score = svm.decision_func_score(testMat) * -1;
            cout<<" the confidence score is : "<<confidence_score<<endl;
            cout<< " the prediction is : "<<pred<<endl;
            if(pred == 1)
            {
                cout<<"Detection: Location ->("<<iter->x<<","<<iter->y<<")"<<endl;
                cout<<"Scale -> {"<<scale<<"} | Confidence Score {"<<confidence_score<<"}"<<endl;
                detect tempDetection;
                tempDetection.x_top_left = iter->x;
                tempDetection.y_top_left = iter->y;
                tempDetection.width_of_detections = (int)(window_size[0]*(pow(downscale,scale)));
                tempDetection.height_of_detections = (int)(window_size[1]*(pow(downscale,scale)));
                tempDetection.confidence_of_detections = confidence_score;
                detections.push_back(tempDetection);
                cd.push_back(detections.back());
            }
        }
        
        dest.release();
        pyrDown(tmp,dest,Size(tmp.cols/2,tmp.rows/2));
        //cout<<"debug-"<<i<<" "<<dest.cols<<" "<<dest.rows<<endl;
        //cout<<window_size[0]<<" "<<window_size[1]<<endl;
        if(dest.cols < window_size[0] || dest.rows < window_size[1])
        {
            break;
        }
        
        tmp.release();
        tmp = dest;
        
        scale += 1;
    }
    
    vector<detect> result;
    nms(detections,result);
    Mat clone = testImg;
    
    if(result.size() == 0)
        cout<<"hehehehe"<<endl;
    for(vector<detect>::iterator iter = result.begin(); iter != result.end(); iter++)
    {
        
        rectangle(clone, Rect(iter->x_top_left, iter->y_top_left, iter->width_of_detections, iter->height_of_detections),(255,0,0),2);
        cout<<iter->confidence_of_detections<<endl;
    }
    
    imshow("test",clone);
    waitKey();
    
    return 0;
}