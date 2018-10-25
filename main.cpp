#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <chrono>

int main(int argc, char* argv[]) {

    double alpha = 2.3;

    cv::setNumThreads(1); // restrict OpenCV to use single thread

    if (argc != 3) {
        std::cout << "Wrong number of arguments! " << std::endl;
        return 1;
    }

    std::string inputFileName = std::string(argv[1]);
    std::string outputFileName = std::string(argv[2]);

    cv::Mat image = cv::imread(inputFileName, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat newImage = cv::Mat::zeros( image.size(), image.type() );

    if (!image.data) {
        std::cout << "Unable to open image!" << std::endl;
        return 2;
    }

    auto start = std::chrono::system_clock::now();

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
                newImage.at<uchar >(y,x) = static_cast<uchar>(alpha*(image.at<uchar>(y,x)));
        }
    }

    auto end = std::chrono::system_clock::now();

    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    cv::imwrite(outputFileName, newImage, compression_params);

    std::chrono::duration<double> diff = end-start;
    std::cout << std::endl << "Complete in " << diff.count() << "s" <<std::endl;

    return 0;
}