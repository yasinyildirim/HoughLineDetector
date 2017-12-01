/***



*/
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