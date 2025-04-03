#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const std::vector<Point> &, Mat&);

void drawAxis(Mat& src, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
    double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));

    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(src, p, q, colour, 1, LINE_AA);

    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(src, p, q, colour, 1, LINE_AA);

    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(src, p, q, colour, 1, LINE_AA);
}

double getOrientation(const std::vector<Point> &pts, Mat &src)
{
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);

    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
    static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

    std::vector<Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
        pca_analysis.eigenvectors.at<double>(i, 1));

        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }

    circle(src, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(src, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(src, cntr, p2, Scalar(255, 255, 0), 5);

    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x);

    return angle;
}


Mat dftProcess(Mat input) {
    Mat I = input;
    Mat padded;
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize( I.cols );
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);

    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    magI += Scalar::all(1);
    log(magI, magI);

    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));
    Mat q1(magI, Rect(cx, 0, cx, cy));
    Mat q2(magI, Rect(0, cy, cx, cy));
    Mat q3(magI, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, NORM_MINMAX);
    return magI;
}

int main()
{
    // TODO: Make this an arg
    std::string image_path = samples::findFile("./data/examples/Cornell_box.png");
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // Default target image
    imshow("Shikov", img);

    enum KeyState {
        greyscale,
        save,
        pca,
        cie,
        dft,
        quit
    };

    std::map<char, KeyState> KeyStates =  {{'g', greyscale}, {'s', save}, {'p', pca}, {'d', dft}, {'c', cie}, {'q', quit}};

    Mat displayImg = img.clone();

    while (true) {
        int k = waitKey(0);

        switch (KeyStates[k]) {
            case greyscale: {
                Mat grey;
                cvtColor(img, grey, COLOR_BGR2GRAY);
                imshow("Spectral Suite", grey);
                displayImg = grey.clone();
                break;
            }
            case save: {
                imwrite("./data/out/out.jpeg", displayImg);
                break;
            }

             case cie: {
                Mat cie;
                cvtColor(img, cie, COLOR_BGR2Lab);
                imshow("Spectral Suite", cie);
                displayImg = cie.clone();
                break;
            }

            case pca: {
                displayImg = img.clone();


                Mat grey;
                cvtColor(displayImg, grey, COLOR_BGR2GRAY);

                Mat binary;
                threshold(grey, binary, 50, 255, THRESH_BINARY | THRESH_OTSU);

                std::vector<std::vector<Point> > contours;
                findContours(binary, contours, RETR_LIST, CHAIN_APPROX_NONE);

                for (size_t i = 0; i < contours.size(); i++)
                {
                    double area = contourArea(contours[i]);
                    if (area < 1e2 || 1e5 < area) continue;

                    drawContours(displayImg, contours, static_cast<int>(i), Scalar(0, 0, 255), 2);
                    getOrientation(contours[i], displayImg);
                 }

                imshow("Spectral Suite", displayImg);
                break;
            }
            case dft: {
                Mat greyImg = imread(image_path, IMREAD_GRAYSCALE);
                displayImg = dftProcess(greyImg);
                imshow("Spectral Suite", displayImg);
                break;
            }
            case quit: {
                return 0;
            }
        }
    }
    return 0;
}
