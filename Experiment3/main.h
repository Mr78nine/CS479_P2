// C++
#include <iostream>
#include <vector>

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
}