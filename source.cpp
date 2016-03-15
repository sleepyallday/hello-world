#include "com_luoluo_pic_algorithm_StripAlgorithm.h"
using namespace cv;		
JNIEXPORT jintArray JNICALL Java_com_luoluo_pic_algorithm_StripAlgorithm_getPoint
  (JNIEnv *env, jobject obj, jintArray data, jint w, jint h)
{
	jint *data2;
	data2=env->GetIntArrayElements(data,false);
	Mat raw_img(h,w,CV_8UC3,(unsigned char*)data2);
	int q[9]={1,2,3,4,5,6,7,8,9};
	jint p[9];
	for(int i=0;i<9;i++){
		p[i] = q[i];
	}
	jintArray ret = env->NewIntArray(9);
	env->SetIntArrayRegion(ret,0,9,p);
	return ret;
}