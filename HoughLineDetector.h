/*#******************************************************************************
 ** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 **
 ** By downloading, copying, installing or using the software you agree to this license (LGPL).
 ** If you do not agree to this license, do not download, install,
 ** copy or use the software.
 **
 ** See COPYING file for license information.
 **
 **  Creation - November 2017
 **      Author: Yasin Yıldırım (yildirimyasi(at)gmail(dot)com), Istanbul, Turkey
 **
*******************************************************************************/
#pragma once
#include <vector>
#include <opencv2\core.hpp>
#include <string>
#include <opencv2\ml.hpp>
#define NCLUSTER_DEFAULT 25
class HoughLineDetector
{
public:
	HoughLineDetector();
	~HoughLineDetector();

	void detect(const cv::Mat& src, std::vector<cv::Vec2f> &lines,
		double rhoResolution, double thetaResolution,
		int voteThreshold, int edgeThreshold = 50);

private:
	//double rhoResolution;
	//double thetaResolution;
	cv::Ptr<cv::ml::EM> expMaxAlgorithm;

};

// Classical Hough Transform
struct LinePolar
{
	float rho;
	float angle;
};
