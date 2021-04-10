// C++
#include <iostream>
#include <vector>
#include "../Common/image.h"
#include "../Eigen/Dense"
#include "../Eigen/StdVector"
#include "../Eigen/Core"
#include <cmath>

// OpenCV
#include <opencv2/opencv.hpp>

#define TRAIN_IMAGE1_PATH "../Images/Training_1.ppm"
#define TRAIN_REF1_PATH "../Images/ref1.ppm"
#define TEST_IMAGE1_PATH "../Images/Training_3.ppm"
#define TEST_REF1_PATH "../Images/ref3.ppm"
#define TEST_IMAGE2_PATH "../Images/Training_6.ppm"
#define TEST_REF2_PATH "../Images/ref6.ppm"

/**
 * @brief Get a row major vector of bgr pixel values of an image
 * 
 * @param image 
 * @return std::vector<cv::Vec3b> 
 */
std::vector<cv::Vec3b> GetPixelVector(cv::Mat& image)
{
  std::vector<cv::Vec3b> ret_vec;
  ret_vec.reserve(image.rows * image.cols);

  // Iterate through the image pixels in a row major fashion
  for (int i = 0; i < image.rows; ++i)
  {
    for (int j = 0; j < image.cols; ++j)
    {
      // Get pixel values at a specific pixel
      ret_vec.push_back(image.at<cv::Vec3b>(i, j));
    }
  }

  return ret_vec;
  
// Part1-Bayes/main.h


typedef struct ChrColors {

    float r;
    float g;
    ChrColors(float R, float G, float B){
        r = R / (R + G + B);
        g = G / (R + G + B);
    }
}chrColors;

typedef struct GaussainParams{
    float r_bar;
    float g_bar;

    Eigen::MatrixXd covMatrix;

    GaussainParams(const std::vector<ChrColors> & colors ) {

        //Initialize Sample Means
        r_bar = g_bar = 0;
        float snum = colors.size();
        for (auto & color : colors)
        {
            r_bar += (color.r/snum);
            g_bar += (color.g/snum);
        }

        //Initialize Cov Matrix
        float covrr = 0;
        float covrg = 0;
        float covgr = 0;
        float covgg = 0;

        for (auto & color : colors)
        {
            covrr += ((color.r - r_bar) * (color.r - r_bar) / (snum-1));
            covrg += ((color.r - r_bar) * (color.g - g_bar) / (snum-1));
            covgr += ((color.g - g_bar) * (color.r - r_bar) / (snum-1));
            covgg += ((color.g - g_bar) * (color.g - g_bar) / (snum-1));
        }

        covMatrix(0,0) = covrr;
        covMatrix(0,1) = covrg;
        covMatrix(1,0) = covgr;
        covMatrix(1,1) = covgg;

    }
}GaussainParams;

bool isFace(ChrColors pixel, GaussainParams g, float threshold)
{
    Eigen::VectorXd observation(2,1);
    Eigen::VectorXd mean(2,1);
    observation(0,0) = pixel.r; //Param order; r then g
    observation(1,0) = pixel.g;
    mean(0,0) = g.r_bar;
    mean(1,0) = g.g_bar;

    auto det = g.covMatrix.determinant();

    auto term1 = 1 / (2*M_PI* sqrt(det));
    auto term2 = -0.5 * (observation - mean).transpose() *
                 g.covMatrix.inverse() * (observation - mean);

    float p =  pow(term1, term2);

    return p > threshold ;
}