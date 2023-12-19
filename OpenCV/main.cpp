#include <iostream>
#include <opencv2/opencv.hpp>

constexpr auto winName = "shock_filter";

void gradient_norm(const cv::Mat& aInput, cv::Mat& aOutput, cv::Mat& aGx,
                   cv::Mat& aGy) {
  cv::Sobel(aInput, aGx, aInput.depth(), 1, 0);
  cv::Sobel(aInput, aGy, aInput.depth(), 0, 1);
  cv::multiply(aGx, aGx, aGx);
  cv::multiply(aGy, aGy, aGy);
  cv::sqrt(aGx + aGy, aOutput);
}

int main(int argc, char** argv) {
  std::string image_name = "text_motion.jpg";
  if (argc > 1) {
    image_name = argv[1];
  }

  cv::Mat source = cv::imread(image_name, cv::IMREAD_COLOR);
  if (source.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  cv::Mat img;
  source.convertTo(img, CV_32F);
  img /= 255.0;

  cv::imshow("input", img);

  cv::namedWindow(winName);

  cv::Mat filtered = img;

  const int steps = 30;
  const float stepSize = 0.25;
  const int maskSize = 9;

  // Buffer images
  cv::Mat gx;
  cv::Mat gy;
  cv::Mat lapl;
  cv::Mat grad;

  for (int i = 0; i < steps; i++) {
    cv::GaussianBlur(filtered, filtered, cv::Size(maskSize, maskSize), 0);
    cv::Laplacian(filtered, lapl, filtered.depth());
    gradient_norm(filtered, grad, gx, gy);
    cv::Mat3f mask(filtered.size());
    mask = 0;
    mask.setTo(-1, lapl < 0);
    mask.setTo(1, lapl > 0);
    cv::multiply(mask * stepSize, grad, mask);
    filtered -= mask;

    cv::imshow(winName, filtered);
    char ch = cv::waitKey(0);
    if (ch == 27) {
      break;
    }
  }

  cv::imshow(winName, filtered);

  while (true) {
    char ch = cv::waitKey(0);
    if (ch == 27) {
      break;
    }
  }

  return EXIT_SUCCESS;
}
