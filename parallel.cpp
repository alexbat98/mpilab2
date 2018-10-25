#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <chrono>
#include <mpi.h>

inline uchar scalePixel(uchar pixel, double alpha) {
    int scaled = static_cast<int>(alpha * (pixel - 128) + 128);

    uchar res = 0;
    if (scaled > 255) {
        res = 255;
    } else if (scaled < 0) {
        res = 0;
    } else {
        res = static_cast<uchar>(scaled);
    }

    return res;
}

int main(int argc, char *argv[]) {

    double alpha = 0.8;

    MPI_Init(&argc, &argv);

    int procId, procCount;
    double startTime = .0;

    int rows = 0, cols = 0;
    int partSize;
    int tail;
    uchar *data = nullptr;
    uchar *newData = nullptr;

    MPI_Comm_rank(MPI_COMM_WORLD, &procId);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    cv::setNumThreads(1); // restrict OpenCV to use single thread

    if (argc < 3 || argc > 4) {
        std::cout << "Wrong number of arguments! " << std::endl;
        return 1;
    }

    const std::string inputFileName = std::string(argv[1]);
    const std::string outputFileName = std::string(argv[2]);
    if (argc == 4)
        alpha = std::stod(argv[3]);

    if (procId == 0) {
        cv::Mat image = cv::imread(inputFileName, CV_LOAD_IMAGE_GRAYSCALE);

        if (!image.data) {
            std::cout << "Unable to open image!" << std::endl;
            return 2;
        }

        rows = image.rows;
        cols = image.cols;

        if (image.isContinuous()) {
            data = image.data;
        } else {
            std::cout << "fail!";
        }
        newData = new uchar[rows * cols];
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    partSize = rows / procCount;
    tail = rows % procCount;

    int *sizes = procId == 0 ? new int[procCount] : nullptr;
    int *displacements = procId == 0 ? new int[procCount] : nullptr;

    if (procId == 0) {
        for (int i = 0; i < procCount - 1; i++) {
            sizes[i] = partSize * cols;
            displacements[i] = i * partSize * cols;
        }
        sizes[procCount - 1] = (partSize + tail) * cols;
        displacements[procCount - 1] = rows * cols - (partSize + tail) * cols;
    }

    int receiveCount =
            procId == procCount - 1 ? (partSize + tail) * cols : partSize * cols;

    auto *receiveBuffer = new uchar[receiveCount];
    auto sendBuffer = new uchar[receiveCount];


    startTime = MPI_Wtime();

    double scatterStart = MPI_Wtime();
    MPI_Scatterv(data, sizes, displacements, MPI_UNSIGNED_CHAR, receiveBuffer, receiveCount,
                 MPI_UNSIGNED_CHAR, 0,
                 MPI_COMM_WORLD);
    double scatterEnd = MPI_Wtime();

    double pstart = MPI_Wtime();
    for (int i = 0; i < partSize*cols; i++) {
//        sendBuffer[i] = static_cast<uchar>(alpha * (receiveBuffer[i] - 128) + 128);
        sendBuffer[i] = scalePixel(receiveBuffer[i], alpha);
    }

    double pend = MPI_Wtime();

    double gatherStart = MPI_Wtime();
    MPI_Gatherv(sendBuffer, receiveCount, MPI_UNSIGNED_CHAR, newData, sizes, displacements,
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    double gatherEnd = MPI_Wtime();

    double endTime = MPI_Wtime();

    if (procId == 0) {
        std::vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        cv::Mat newImage(rows, cols, CV_8U, newData);

        cv::imwrite(outputFileName, newImage, compression_params);

        std::cout << std::endl << "Complete in " << endTime - startTime << "s" <<std::endl;
    }

//    std::cout << std::endl << "Scatter on " << procId << " complete in " << scatterEnd - scatterStart << "s" <<std::endl;
//    std::cout << std::endl << "Processing on " << procId << " complete in " << pend - pstart << "s" <<std::endl;
//    std::cout << std::endl << "Processing on " << procId << " complete in " << gatherEnd - gatherStart << "s" <<std::endl;

    MPI_Finalize();

    return 0;
}