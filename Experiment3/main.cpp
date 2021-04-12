// Experiment3/main.cpp
#include "main.h"

int main(int argc, char **argv)
{
  std::cout << "Experiment 3" << std::endl;

  // Create gaussian using the training data
  cv::Mat training_image = cv::imread(TRAIN_IMAGE1_PATH);
  cv::Mat training_image_labels = cv::imread(TRAIN_REF1_PATH);
  std::vector<ChrColors> train_chr_colors = GetChrColors(training_image);
  std::vector<bool> train_labels = GetLabels(training_image_labels);
  GaussianParams gp(GetFaceChrColors(train_chr_colors, train_labels));
  gp.Print();

  // // Verify code correctness on training image
  // cv::Mat colored_image = ColorizeByLabels(training_image, train_labels);
  // cv::imshow("Colorized image", colored_image);
  // cv::waitKey(0);

  // Classify one image
  cv::Mat test_image = cv::imread(TEST_IMAGE1_PATH);
  cv::Mat test_image_labels = cv::imread(TEST_REF1_PATH);
  std::vector<bool> test_actual_labels = GetLabels(test_image_labels);
  std::vector<ChrColors> test_chr_colors = GetChrColors(test_image);

  float best_threshold = GetBestThreshold(test_chr_colors, test_actual_labels, gp, 0.0, 0.5, 3);
  std::vector<bool> test_calculated_labels = ClassifyChrColors(test_chr_colors, gp, best_threshold);
  cv::Mat colored_image = ColorizeByLabels(test_image, test_calculated_labels);
  cv::imwrite("colored_image.png", colored_image);

  // //Get our gaussian
  // cv::Mat train1 = cv::imread(TRAIN_IMAGE1_PATH);
  // cv::Mat test1 = cv::imread(TEST_IMAGE1_PATH);
  // std::vector<bool> labels = get_labels(TEST_IMAGE1_PATH);
  // std::vector<ChrColors> testChromatic;

  // auto testPixels = GetPixelVector(test1);
  // for (auto &pixel : testPixels) {
  //     ChrColors chrom(pixel[2], pixel[1], pixel[0]);
  //     testChromatic.emplace_back(chrom);
  //   }
  // GaussainParams g2 = get_gaussian(testChromatic, labels);
  // g2.print();

  // try_different_thresholds(TRAIN_IMAGE1_PATH, TEST_IMAGE1_PATH,g,0,3, 3.0/20 );
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
