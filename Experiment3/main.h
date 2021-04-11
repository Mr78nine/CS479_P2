// C++
#include <iostream>
#include <vector>
#include "../Eigen/Dense"
#include "../Eigen/StdVector"
#include "../Eigen/Core"
#include <cmath>
#include <tuple>

// OpenCV
#include <opencv2/opencv.hpp>

#define TRAIN_IMAGE1_PATH "../Images/Training_1.ppm"
#define TRAIN_REF1_PATH "../Images/ref1.ppm"
#define TEST_IMAGE1_PATH "../Images/Training_3.ppm"
#define TEST_REF1_PATH "../Images/ref3.ppm"
#define TEST_IMAGE2_PATH "../Images/Training_6.ppm"
#define TEST_REF2_PATH "../Images/ref6.ppm"

typedef struct ChrColors {

    float r;
    float g;
    ChrColors(float R, float G, float B){
        if ((R+G+B) == 0)
        {
            r = 0;
            g = 0;
        }
        else
        {
            r = R / (R + G + B);
            g = G / (R + G + B);
        }
    }
}ChrColors;

typedef struct GaussainParams{
    float r_bar;
    float g_bar;
    float c; //Normalizing constant
    Eigen::Matrix2d covMatrix;

    GaussainParams(const std::vector<ChrColors> & colors ) {

        //Initialize Sample Means
        r_bar = g_bar = 0.0f;
        float r = 0, g = 0;
        float snum = colors.size();
        for (auto & color : colors)
        {
            r += color.r;
            g += color.g;
        }
        r_bar = r/snum;
        g_bar = g/snum;
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

        //Initialize c
        auto det = covMatrix.determinant();
        c = 1 / (2*M_PI* sqrt(det));

    }
}GaussianParams;


std::vector<bool> classify(const char* trainDir, const char* testDir, float threshold, const char* fileName);
bool isFace(ChrColors pixel, const GaussianParams &g, float threshold);

float try_different_thresholds(const char* trainDir, const char* testDir, GaussianParams g, float threshBot = 0, float threshTop = 3, float threshSteps=0.1, const char* fileName = "threshold_vs_accuracy");
std::vector<bool> classify(const char* trainDir, const char* testDir,  GaussianParams g,float threshold, const char* fileName = "threshold_vs_accuracy");
float get_accuracy(std::vector<bool> classifications,  std::vector<bool> labels);
std::pair<long,long> get_fp_fn(std::vector<bool> classifications, std::vector<bool> labels);
std::vector<bool> get_labels(const char* imgDir);
GaussianParams get_gaussian(std::vector<ChrColors> testPixels, std::vector<bool> labels) ;
/**
 * @brief Get a row major vector of bgr pixel values of an image
 * 
 * @param image 
 * @return std::vector<cv::Vec3b> 
 */

std::vector<cv::Vec3b> GetPixelVector(cv::Mat& image) {
    std::vector<cv::Vec3b> ret_vec;
    ret_vec.reserve(image.rows * image.cols);

    // Iterate through the image pixels in a row major fashion
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            // Get pixel values at a specific pixel
            ret_vec.push_back(image.at<cv::Vec3b>(i, j));
        }
    }

    return ret_vec;
}
// Part1-Bayes/main.h




bool isFace(ChrColors pixel, const GaussianParams &g, float threshold)
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
/**
 * @brief: Find an acceptable threshold by training on TRAIN_IMAGE1_PATH and comparing results with
 *         TRAIN_REF1_PATH
 * @output: Should return the best performing threshold and also write out results of different thresholds to a file
 *          called "threshold_performances.txt". Format of file is:
 *          Threshold_Value  Accuracy
 */
float try_different_thresholds(const char* trainDir, const char* testDir, GaussianParams g, float threshBot, float threshTop, float threshSteps, const char* fileName)
{   //Open file
    std::ofstream file;
    char sname[80] = {'\0'};
    sprintf(sname, "%s_%d" , fileName, rand());
    file.open(sname, std::ios::trunc);
    file << trainDir << testDir << std::endl;
    file << "Treshold" << "" << "FP" << " " << "FN" << " " << "Accuracy" << std::endl;

    //Start with an initial guess of threshold
    float threshold = threshBot;
    float bestThreshold = threshold;
    long fp = LONG_MAX, fn = LONG_MAX;
    //Get image vector

    for (threshold; threshold < threshTop; threshold += threshSteps)
    {
        std::cout << "Trying threshold " << threshold << "\n";
        std::vector<bool> classifications = classify(trainDir, testDir,g,threshold);
        std::vector<bool> labels = get_labels(testDir);

        //Get fp and fn values
        std::pair<long,long> fpfn = get_fp_fn(classifications, labels);
        long newfp = std::get<0>(fpfn);
        long newfn = std::get<1>(fpfn);

        if (newfp < fp) fp = newfp;
        if (newfn < fn) fn = newfn;

        //Get overall accuracy
        float accuracy = (float(labels.size()) -float(newfp + newfn))/float(labels.size());

        //Record threshold roc/accuracy
        file << threshold << " " << newfp << " " << newfn << " " << accuracy << std::endl;

    }

return bestThreshold;
}

std::vector<bool> classify(const char* trainDir, const char* testDir,  GaussianParams g,float threshold, const char* fileName){
    //Get image vectors
    cv::Mat train_image = cv::imread(trainDir);
    cv::Mat test_image = cv::imread(testDir);
    auto trainPixels = GetPixelVector(train_image);
    auto testPixels = GetPixelVector(test_image);


    std::vector<bool> classifications;
    std::vector<bool> labels;

    //Get correct labels first
    labels = get_labels(testDir);

    //Perform classification
    //First, transform our training pixels into the chromatic color space
    std::vector<ChrColors> trainChromatic; //RGB order
    for (auto &pixel : trainPixels) {
        ChrColors chrom(pixel[2], pixel[1], pixel[0]);
        trainChromatic.emplace_back(chrom);
    }


    //Use gaussian with threshold to classify
    for (auto &pixel : trainChromatic) {
        classifications.emplace_back(isFace(pixel, g, threshold));
    }

    return classifications;
}

float get_accuracy(std::vector<bool> classifications,  std::vector<bool> labels) {

    float correctClassifications = 0;
    float sampleNum = labels.size();
    for (long i = 0; i < sampleNum  ; i++ )
    {
        if (classifications[i] == labels[i]) correctClassifications += 1;
    }

    return correctClassifications / sampleNum;

}

std::pair<long,long> get_fp_fn(std::vector<bool> classifications, std::vector<bool> labels)
{
    long fp = 0, fn = 0;

    float sampleNum = labels.size();
    for (long i = 0; i < sampleNum; i++)
    {
        bool c = classifications[i];
        bool l = labels[i];

        //If we're right, no fn/fp
        if ( c == l) continue;
        else if (c == true && l == false) fp += 1;
        else if (c == false && l == true) fn += 1;
    }

    return std::make_pair(fp,fn);
}

std::vector<bool> get_labels(const char* imgDir)

{   cv::Mat test_image = cv::imread(imgDir);
    auto testPixels = GetPixelVector(test_image);

    std::vector<bool> labels;

    for (auto &pixel : testPixels) {
        if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) //Black means no face
        {
            labels.emplace_back(false);
        }
        else {
            labels.emplace_back(true);
        }
    }

    return labels;
}

GaussianParams get_gaussian(std::vector<ChrColors> testPixels, std::vector<bool> labels) {

    //Only use face pixels for the gaussian
    std::vector<ChrColors> faceChromatic;
    for (int i = 0; i < testPixels.size(); i++) {
        auto pixel = testPixels[i];
        if (labels[i] == true) {
            faceChromatic.emplace_back(pixel);
        }
    }

    GaussianParams g(faceChromatic);
    return g;

}