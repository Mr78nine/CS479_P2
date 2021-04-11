// Experiment3/main.cpp
#include "main.h"

int main(int argc, char **argv)
{
  std::cout << "Experiment 3" << std::endl;

  //Get our gaussian
  cv::Mat train1 = cv::imread(TRAIN_IMAGE1_PATH);
  cv::Mat test1 = cv::imread(TEST_IMAGE1_PATH);
  std::vector<bool> labels = get_labels(TEST_IMAGE1_PATH);
  std::vector<ChrColors> testChromatic;

  auto testPixels = GetPixelVector(test1);
  for (auto &pixel : testPixels) {
      ChrColors chrom(pixel[2], pixel[1], pixel[0]);
      testChromatic.emplace_back(chrom);
    }
  GaussainParams g = get_gaussian(testChromatic, labels);
\

  try_different_thresholds(TRAIN_IMAGE1_PATH, TEST_IMAGE1_PATH,g,0,3, 3/20 );
  return 0;
}

//  int r = 498;
//  int c = 1606;

//  // Read Images
//  cv::Mat train_image_1 = cv::imread(TRAIN_IMAGE1_PATH);
//  cv::Mat train_ref_1 = cv::imread(TRAIN_REF1_PATH);
//  cv::Mat test_image_1 = cv::imread(TEST_IMAGE1_PATH);
//  cv::Mat test_ref_1 = cv::imread(TEST_REF1_PATH);
//  cv::Mat test_image_2 = cv::imread(TEST_IMAGE2_PATH);
//  cv::Mat test_ref_2 = cv::imread(TEST_REF2_PATH);

// Visualize images in a single window. Images will advance on pressing any key
//  cv::imshow("Image", train_image_1);
//  cv::waitKey(0);
//  cv::imshow("Image", train_ref_1);
//  cv::waitKey(0);
//  cv::imshow("Image", test_image_1);
//  cv::waitKey(0);
//  cv::imshow("Image", test_ref_1);
//  cv::waitKey(0);
//  cv::imshow("Image", test_image_2);
//  cv::waitKey(0);
//  cv::imshow("Image", test_ref_2);
//  cv::waitKey(0);

//  std::vector<cv::Vec3b> pixel_vector = GetPixelVector(train_ref_1);
//  // Get the bgr pixel values at (r, c)
//  cv::Vec3b bgr_pixel = pixel_vector[r * train_ref_1.cols + c];
//  std::cout << "BGR at (" << r << ", " << c << "): " << int(bgr_pixel[0]) << ", " << int(bgr_pixel[1]) << ", " << int(bgr_pixel[2]) << std::endl;
