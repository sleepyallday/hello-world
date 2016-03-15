#include "com_luoluo_pic_algorithm_StripAlgorithm.h"
#define DISTANCE(a,b)(sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)))
using namespace cv;


int Hue_judgement(Mat input,Mat H_judge_binary_map){
	
		Mat h=input.clone();
		int counter[181];
		memset(counter,0,sizeof(counter));
		MatIterator_<uchar> it,end;
		for(it=h.begin<uchar>(),end=h.end<uchar>();it!=end;++it){
			counter[(*it)]++;
		}

		int big=1;
		int sum=0;
		for(int j=1;j<141;j++){
			int tmp_sum=0;
			for(int i=j;i<j+41;i++){
				tmp_sum+=counter[i];
			}
			if(sum<tmp_sum){
				sum=tmp_sum;
				big=j;
			}
		}

		for(it=h.begin<uchar>(),end=h.end<uchar>();it!=end;++it){
			if((*it)>=big&&(*it)<=big+40)(*it)=255;
			else (*it)=0;
		}
		
		Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
		erode(h,h,element);
		Mat element1 = getStructuringElement(MORPH_RECT, Size(7,7));
		dilate(h,h,element1);
		dilate(h,h,element1);
		h.copyTo(H_judge_binary_map);
		return 0;
}

int Value_judgement(Mat input,Mat V_judge_binary_map){
	#define BORDER_WIDTH 50
	Mat v=input.clone();
	Mat tmp_count_map;
	Mat count_map;
	Mat	int_map;
	threshold(v,tmp_count_map,0.5,1,0);
	integral(v, int_map);
	integral(tmp_count_map,count_map);

	for (int y = 0; y < v.rows; y++)
	{
		for (int x = 0; x < v.cols; x++)
		{
			int tlx = x - BORDER_WIDTH, tly = y - BORDER_WIDTH, brx = x + BORDER_WIDTH, bry = y + BORDER_WIDTH;
			if (tlx < 0)tlx = 0;
			if (tly < 0)tly = 0;
			if (brx > int_map.cols - 1) brx = int_map.cols - 1;
			if (bry > int_map.rows - 1) bry = int_map.rows - 1;

			int sum = int_map.at<int>(bry, brx) + int_map.at<int>(tly, tlx)
				- int_map.at<int>(bry, tlx) - int_map.at<int>(tly, brx);
			int area=1+count_map.at<int>(bry, brx) + count_map.at<int>(tly, tlx)
				- count_map.at<int>(bry, tlx) - count_map.at<int>(tly, brx);
			if(v.at<uchar>(y,x)>(((uchar)(sum / area)-10)>10?((uchar)(sum / area)-10):10)&&v.at<uchar>(y,x)<250)v.at<uchar>(y,x)=255;
			else v.at<uchar>(y,x)=0;
		}
	}
	v.copyTo(V_judge_binary_map);
	return 0;
}

int R_judgement(Mat R,Mat R_judge_binary_map){
		MatIterator_<uchar> it,end;
		for(it=R.begin<uchar>(),end=R.end<uchar>();it!=end;++it){
			if((*it)>20&&(*it)<240)(*it)=255;
			else (*it)=0;
		}
		R.copyTo(R_judge_binary_map);
		return 0;
}

int find_centers(Mat input,vector<Point>& centers){
	vector<Point> output_centers;
	for (int morphsize = 6; morphsize <= 50; morphsize+=2){
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morphsize + 1, 2 * morphsize + 1), Point(morphsize, morphsize));
		Mat morphRet;

		erode(input.clone(), morphRet, element);
		vector<vector<Point>> conts;
		findContours(morphRet, conts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		for (int idx = 0; idx < conts.size(); idx++)
		{
			int  contArea = (int)contourArea(conts[idx]);
			if (contArea < 30)
			{
				Moments mu;
				mu = moments(conts[idx], false);
				Point2f center = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
				output_centers.push_back(Point(center));
			}
		}
	}
	centers.push_back(output_centers[0]);
	for(int i=1;i<output_centers.size();i++){
		bool repet=false;
		for(int j=0;j<centers.size();j++){
			if(DISTANCE(output_centers[i],centers[j])<25){
				repet=true;
				break;
			}
		}
		if(!repet)centers.push_back(output_centers[i]);
	}
	return 0;
}


//JNIEXPORT jintArray JNICALL Java_com_luoluo_pic_algorithm_StripAlgorithm_getPoint
//  (JNIEnv *env, jobject obj, jintArray data, jint w, jint h)
//{
//	jint *data2;
//	data2=env->GetIntArrayElements(data,false);
//	Mat raw_img(h,w,CV_8UC3,(unsigned char*)data2);
//	int q[9]={1,2,3,4,5,6,7,8,9};
//	jint p[9];
//	for(int i=0;i<9;i++){
//		p[i] = q[i];
//	}
//	jintArray ret = env->NewIntArray(9);
//	env->SetIntArrayRegion(ret,0,9,p);
//	return ret;
//}

JNIEXPORT jintArray JNICALL Java_com_luoluo_pic_algorithm_StripAlgorithm_getPoint
  (JNIEnv *env, jobject obj, jintArray data, jint w, jint h)
{
	jint *data2;
	data2=env->GetIntArrayElements(data,false);
	Mat raw_img(h,w,CV_8UC3,(unsigned char*)data2);
	Mat src=raw_img.clone();

	if(src.rows>2000&&src.cols>2000)resize(src,src,Size(src.cols/2,src.rows/2));
	if((src.rows<600&&src.cols<1000)||(src.rows<1000&&src.cols<600))resize(src,src,src.size()*2);
	Mat out=src.clone();


	Mat hsv, _h, _s, _v;
	vector<Mat> hsvMats;

	pyrMeanShiftFiltering(src, src, 5, 5, 3);


	Mat RGB,R,G,B;
	vector<Mat> RGBMats;
	split(src,RGBMats);
	R=RGBMats[0];
	Mat R_judge_binary_map = Mat(R.size(), CV_8UC1, Scalar(0));
	R_judgement(R,R_judge_binary_map);
	
	cvtColor(src, hsv, CV_RGB2HSV);
	split(hsv, hsvMats);
	_h = hsvMats[0];
	_s = hsvMats[1];
	_v = hsvMats[2];

	Mat H_judge_binary_map = Mat(src.size(), CV_8UC1, Scalar(0));
	Mat V_judge_binary_map = Mat(src.size(), CV_8UC1, Scalar(0));

	Hue_judgement(_h,H_judge_binary_map);
    Value_judgement(_v,V_judge_binary_map);
	Mat bin_map;
	bitwise_and(R_judge_binary_map,V_judge_binary_map,bin_map);
	Mat element1 = getStructuringElement(MORPH_RECT, Size(5,5));
	dilate(bin_map,bin_map,element1);
	vector<Point> centers;
	find_centers(bin_map,centers);

	int radius=5; 
	if(centers.size()>14){
		for(int i=centers.size()/2-5;i<=centers.size()/2+5;i++){
			vector<int> distance;
			int small_dis=1000,small_count=0;
			for(int j=0;j<centers.size();j++){
				if(i==j) continue;
				distance.push_back(DISTANCE(centers[i],centers[j]));
			}
			for(int j=0;j<5;j++){
				for(int k=0;k<distance.size();k++){
					if(small_dis>=distance[k]){
						small_dis=distance[k];
						small_count=k;
					}
				}
				radius+=small_dis;
				small_dis=1000;
				distance[small_count]=1000;
			}
		}
		radius=(radius/100)*0.3;
	}
		for (int i = 0; i < centers.size(); i++)
	{
		circle(out, centers[i], radius, Scalar(0, 255, 0), -1);
	}
	
	int num_centers=centers.size();
	jint *outarray=new jint[num_centers];
	int counter=0;
	for(int i=0;i<num_centers;i++){
		outarray[counter++]=centers[i].x;
		outarray[counter++]=centers[i].y;
		outarray[counter++]=radius;
	}

	jintArray ret = env->NewIntArray(num_centers);
	env->SetIntArrayRegion(ret,0,num_centers,outarray);
	delete []outarray;
	return ret;
}