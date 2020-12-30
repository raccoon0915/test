#include<opencv2/opencv.hpp>

using namespace cv;

extern void guassain_conv(const Mat*,Mat*,double);
int main(int argc,char** argv){
	Mat img=imread(argv[2]);
	Mat result=imread(argv[2]);
	guassain_conv(&img,&result,atof(argv[1]));
	imwrite(argv[3],result);
	return 0;
}
