#include "HoughLineDetector.h"
#include <opencv2\imgproc.hpp>
#include <opencv2\ml.hpp>
//TODO: remove highgui
#include <opencv2\highgui.hpp>
#include "DebugUtils.h"

HoughLineDetector::HoughLineDetector()
{
	expMaxAlgorithm = cv::ml::EM::create();
	expMaxAlgorithm->setClustersNumber(NCLUSTER_DEFAULT);
	cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.02);
	expMaxAlgorithm->setTermCriteria(tc);
}


HoughLineDetector::~HoughLineDetector()
{
	
}

/**

inspired from opencv's HoughLines() method.
*/
void HoughLineDetector::detect(const cv::Mat& src, std::vector<cv::Vec2f> &lines,
	double rhoResolution, double thetaResolution, int voteThreshold, int edgeThreshold) {
	
	//min and max theta variables can be input parameters as well.
	int minTheta = 0, maxTheta = CV_PI;
	float rhoStep = 1 / rhoResolution;
	
	
	CV_Assert(src.depth() == CV_8U);

	int channels = src.channels();
	CV_Assert(channels == 1);
	int nRows = src.rows;
	int nCols = src.cols * channels;

	int numAngle = cvRound((maxTheta - minTheta) / thetaResolution);
	int numRho = cvRound(((nRows + nCols) * 2 + 1) / rhoStep);
	cv::Mat houghSpaceAccum = cv::Mat::zeros(numAngle+2, numRho+2, CV_32SC1);
	cv::AutoBuffer<float> _sinTable(numAngle);
	cv::AutoBuffer<float> _cosTable(numAngle);
	float* sinTable = _sinTable, *cosTable = _cosTable;
	float currAng = static_cast<float>(minTheta);
	for (int n = 0;
		n < numAngle; currAng += thetaResolution, ++n)
	{
		sinTable[n] = (float)(sin(currAng) * rhoStep);
		cosTable[n] = (float)(cos(currAng) * rhoStep);
	}

	//rho = x*cos(theta) + y*sin(theta)
	const uchar* p;
	//step 1: calculate hough space accumulator
	for (int i = 0; i < nRows; ++i)
	{
		p =  src.ptr<uchar>(i) ;
		for (int j = 0; j < nCols; ++j)
		{
			//p[j] ;
			//only consider the edges that are strong enough 
			if (p[j] > edgeThreshold) {
				for (int k = 0; k < numAngle; ++k) {
					int rhoIdx =  cvRound( j*cosTable[k] + i*sinTable[k]) ;
					
					rhoIdx += (numRho - 1) / 2;
					int& val = houghSpaceAccum.at<int>(k+1, rhoIdx + 1);
					//val += p[j];
					++val;
				}
			}
		}
	}
	
	//houghSpaceAccum.convertTo(houghSpaceAccum, CV_32F);
	//cv::threshold(houghSpaceAccum, houghSpaceAccum, voteThreshold, 500, CV_THRESH_TOZERO);
	
	//cv::dilate(houghSpaceAccum, houghSpaceAccum, cv::Mat(), cv::Point(-1,-1));
	//cv::dilate(houghSpaceAccum, houghSpaceAccum, 4cv::Mat(), cv::Point(-1, -1));
	//cv::kmeans();
	
	cv::Mat clsSamples(0,0, CV_64F);
	double pattArray[3];
	//const float* accumfPtr;
	const int* accumiPtr;
	for (int i = 0; i < houghSpaceAccum.rows; ++i)
	{
		accumiPtr = houghSpaceAccum.ptr<int>(i);
		for (int j = 0; j < houghSpaceAccum.cols; ++j)
		{
			//p[j] ;
			//only consider the high scores 
			if (accumiPtr[j] > voteThreshold) {
				//cv::Mat pat = (cv::Mat_<double>(1, 3) << j, i, accumiPtr[j]);
				pattArray[0] = static_cast<double>(j);
				pattArray[1] = static_cast<double>(i);
				pattArray[2] = static_cast<double>(accumiPtr[j]);
				cv::Mat pat(1, 3, CV_64FC1, pattArray);
				clsSamples.push_back(pat);
			}
		}
	}
	cv::Mat gmmMeans;
	if (clsSamples.rows > expMaxAlgorithm->getClustersNumber()) {
		expMaxAlgorithm->trainEM(clsSamples);
		gmmMeans = expMaxAlgorithm->getMeans();
		std::cout << DebugUtils::type2str(gmmMeans.type()) <<std::endl;
		const double* gmmMeansPtr;
		for (int i = 0; i < gmmMeans.rows; ++i) {
			gmmMeansPtr = gmmMeans.ptr<double>(i);
			LinePolar line;
			line.rho = static_cast<float>(gmmMeansPtr[0] - ((numRho + 1) / 2) ) * rhoResolution;
			line.angle = static_cast<float>(minTheta) +  (gmmMeansPtr[1] - 1) * thetaResolution;
			lines.push_back(cv::Vec2f( line.rho, line.angle) );
		}
	}
	
	
	//Below code is for visualisation purposes.
	double maxVal, minVal;
	cv::minMaxIdx(houghSpaceAccum, &minVal, &maxVal);
	houghSpaceAccum -= (float)minVal;
	houghSpaceAccum.convertTo(houghSpaceAccum, CV_8UC1, 255.0 / (maxVal - minVal), 0);
	
	cv::imshow("houghSpaceAccum", houghSpaceAccum);
	//assuming waitKey is called after. 
	
}