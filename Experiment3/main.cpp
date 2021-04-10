// Experiment3/main.cpp
#include "main.h"

int main(int argc, char **argv)
{
  std::cout << "Experiment 3" << std::endl;

  // Read Images
  cv::Mat train_image_1 = cv::imread(TRAIN_IMAGE1_PATH);
  cv::Mat train_ref_1 = cv::imread(TRAIN_REF1_PATH);
  cv::Mat test_image_1 = cv::imread(TEST_IMAGE1_PATH);
  cv::Mat test_ref_1 = cv::imread(TEST_REF1_PATH);
  cv::Mat test_image_2 = cv::imread(TEST_IMAGE2_PATH);
  cv::Mat test_ref_2 = cv::imread(TEST_REF2_PATH);

  // Visualize images in a single window. Images will advance on pressing any key
  cv::imshow("Image", train_image_1);
  cv::waitKey(0);
  cv::imshow("Image", train_ref_1);
  cv::waitKey(0);
  cv::imshow("Image", test_image_1);
  cv::waitKey(0);
  cv::imshow("Image", test_ref_1);
  cv::waitKey(0);
  cv::imshow("Image", test_image_2);
  cv::waitKey(0);
  cv::imshow("Image", test_ref_2);
  cv::waitKey(0);

  // Iterate through the image pixels
  for (int i = 0; i < train_ref_1.rows; ++i)
  {
    for (int j = 0; j < train_ref_1.cols; ++j)
    {
      // Get pixel values at a specific pixel
      cv::Vec3b bgr_pixel = train_ref_1.at<cv::Vec3b>(i, j);
      std::cout << "Value at (" << i << ", " << j << "): (" << int(bgr_pixel[0]) << ", " << int(bgr_pixel[1]) << ", " << int(bgr_pixel[2]) << ")" << std::endl;
    }
  }

  return 0;
}