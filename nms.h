# pragma once

#include <vector>
#include <algorithm>

using namespace std;

struct detect
{
    int x_top_left;
    int y_top_left;
    int width_of_detections;
    int height_of_detections;
    float confidence_of_detections;
};

float overlapping_area(detect detection_1, detect detection_2)
{
    /*
        Function to calculate overlapping area.
        'detection_1' and 'detection_2' are two detections whose 
        area of overlap needs to found out.
        
        Each detection is a struct of type "detect".
        
        The function returns a calue between 0 and 1, which represents
        the area of overlap.
        0 is no overlap and 1 is complete overlap.
    */
    
    // calculate the x-y coordinate of the rectangles
    int x1_tl = detection_1.x_top_left;
    int x2_tl = detection_2.x_top_left;
    int x1_br = detection_1.x_top_left + detection_1.width_of_detections;
    int x2_br = detection_2.x_top_left + detection_2.width_of_detections;
    int y1_tl = detection_1.y_top_left;
    int y2_tl = detection_2.y_top_left;
    int y1_br = detection_1.y_top_left + detection_1.height_of_detections;
    int y2_br = detection_2.y_top_left + detection_2.height_of_detections;
    
    // calculate the overlapping area
    
    int x_overlap = max(0, min(x1_br,x2_br) - max(x1_tl,x2_tl));
    int y_overlap = max(0, min(y1_br,y2_br) - max(y1_tl,y2_tl));
    int overlap_area = x_overlap * y_overlap;
    int area1 = detection_1.width_of_detections * detection_1.height_of_detections;
    int area2 = detection_2.width_of_detections * detection_2.height_of_detections;
    int total_area = area1 + area2 - overlap_area;
    
    return overlap_area / (float)(total_area);
}

bool cmp(detect& obj1, detect& obj2)
{
    return obj1.confidence_of_detections > obj2.confidence_of_detections;
}

void nms(vector<detect>& detections, vector<detect>& result, float threshold = 0.5)
{
    /*
        This function performs Non-Maxima Supression.
        'detections' consists of a list of detections.
        Each detection is in the format of:
        @ x-top-left
        @ y-top-left
        @ width-of-detection
        @ height_of_detection 
        @ confidence_of_detections
        
        If the area of overlap is greater than the 'threshold',
        the area with the lower confidence score is removed.
        The output is a list of detections.
    */
    
    
    if(detections.size() != 0)
    {
        sort(detections.begin(), detections.end(), cmp);
    
        result.push_back(detections[0]);
    
        detections.erase(detections.begin());
    
        vector<detect>::iterator iter = detections.begin();
        for(;iter != detections.end();iter++)
        {
            for(vector<detect>::iterator iter1 = result.begin(); iter1 != result.end(); iter++)
            {
                if( overlapping_area(*iter, *iter1) > threshold )
                {
                    detections.erase(iter);
                    break;
                }
                else
                {
                    result.push_back(*iter);
                    detections.erase(iter);
                }
            }
        }
    }
}