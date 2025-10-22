#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
// No ximgproc dependency needed
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

class AdvancedLeafVeinDetector {
private:
    // Parameters for different algorithms
    struct AlgorithmParams {
        // Sobel parameters
        int sobel_ksize = 3;
        double sobel_scale = 1.0;
        double sobel_delta = 0.0;
        
        // Prewitt parameters (implemented as custom kernels)
        Mat prewitt_x = (Mat_<float>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
        Mat prewitt_y = (Mat_<float>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
        
        // LoG parameters
        int log_ksize = 5;
        double log_sigma = 1.0;
        
        // Canny parameters
        double canny_low_thresh = 50.0;
        double canny_high_thresh = 150.0;
        int canny_aperture = 3;
        
        // Green detection parameters
        Scalar hsv_lower_green = Scalar(35, 40, 40);
        Scalar hsv_upper_green = Scalar(85, 255, 255);
        
        // Morphological parameters
        int morph_kernel_size = 3;
        int skeleton_iterations = 10;
    };
    
    AlgorithmParams params;
    
public:
    AdvancedLeafVeinDetector() {}
    
    // Main function to detect if image contains a leaf
    bool isLeafImage(const Mat& input) {
        Mat hsv, mask;
        cvtColor(input, hsv, COLOR_BGR2HSV);
        
        // Create mask for green colors
        inRange(hsv, params.hsv_lower_green, params.hsv_upper_green, mask);
        
        // Calculate the percentage of green pixels
        int total_pixels = mask.rows * mask.cols;
        int green_pixels = countNonZero(mask);
        double green_percentage = (double)green_pixels / total_pixels;
        
        // Additional shape analysis
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) return false;
        
        // Find the largest contour (assumed to be the leaf)
        double max_area = 0;
        int max_contour_idx = 0;
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_contour_idx = i;
            }
        }
        
        // Check if the largest contour has leaf-like properties
        double perimeter = arcLength(contours[max_contour_idx], true);
        double circularity = 4 * CV_PI * max_area / (perimeter * perimeter);
        
        // Leaf detection criteria
        bool is_leaf = (green_percentage > 0.15) && // At least 15% green
                      (max_area > 1000) &&          // Minimum area
                      (circularity < 0.85) &&       // Not too circular (leaves are elongated)
                      (circularity > 0.1);          // Not too irregular
        
        return is_leaf;
    }
    
    // Sobel edge detection
    Mat applySobel(const Mat& gray) {
        Mat grad_x, grad_y, abs_grad_x, abs_grad_y, sobel_combined;
        
        Sobel(gray, grad_x, CV_16S, 1, 0, params.sobel_ksize, 
              params.sobel_scale, params.sobel_delta, BORDER_DEFAULT);
        Sobel(gray, grad_y, CV_16S, 0, 1, params.sobel_ksize, 
              params.sobel_scale, params.sobel_delta, BORDER_DEFAULT);
        
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_combined);
        
        return sobel_combined;
    }
    
    // Prewitt edge detection
    Mat applyPrewitt(const Mat& gray) {
        Mat prewitt_x_result, prewitt_y_result, abs_prewitt_x, abs_prewitt_y, prewitt_combined;
        
        filter2D(gray, prewitt_x_result, CV_16S, params.prewitt_x);
        filter2D(gray, prewitt_y_result, CV_16S, params.prewitt_y);
        
        convertScaleAbs(prewitt_x_result, abs_prewitt_x);
        convertScaleAbs(prewitt_y_result, abs_prewitt_y);
        
        addWeighted(abs_prewitt_x, 0.5, abs_prewitt_y, 0.5, 0, prewitt_combined);
        
        return prewitt_combined;
    }
    
    // Laplacian of Gaussian (LoG) edge detection
    Mat applyLoG(const Mat& gray) {
        Mat blurred, log_result, abs_log;
        
        GaussianBlur(gray, blurred, Size(params.log_ksize, params.log_ksize), 
                     params.log_sigma, params.log_sigma);
        Laplacian(blurred, log_result, CV_16S, params.log_ksize);
        convertScaleAbs(log_result, abs_log);
        
        return abs_log;
    }
    
    // Canny edge detection
    Mat applyCanny(const Mat& gray) {
        Mat canny_result;
        Canny(gray, canny_result, params.canny_low_thresh, 
              params.canny_high_thresh, params.canny_aperture);
        return canny_result;
    }
    
    // Combined edge detection algorithm
    Mat combineEdgeDetectors(const Mat& gray) {
        Mat sobel = applySobel(gray);
        Mat prewitt = applyPrewitt(gray);
        Mat log_edges = applyLoG(gray);
        Mat canny = applyCanny(gray);
        
        // Normalize all edge maps to same range
        Mat sobel_norm, prewitt_norm, log_norm, canny_norm;
        normalize(sobel, sobel_norm, 0, 255, NORM_MINMAX, CV_8UC1);
        normalize(prewitt, prewitt_norm, 0, 255, NORM_MINMAX, CV_8UC1);
        normalize(log_edges, log_norm, 0, 255, NORM_MINMAX, CV_8UC1);
        normalize(canny, canny_norm, 0, 255, NORM_MINMAX, CV_8UC1);
        
        // Combine using weighted average
        Mat combined;
        addWeighted(sobel_norm, 0.25, prewitt_norm, 0.25, 0, combined);
        addWeighted(combined, 0.5, log_norm, 0.25, 0, combined);
        addWeighted(combined, 0.75, canny_norm, 0.25, 0, combined);
        
        return combined;
    }
    
    // Enhanced vein detection with directional sensitivity
    Mat detectVeinsAllDirections(const Mat& input) {
        Mat gray;
        if (input.channels() == 3) {
            cvtColor(input, gray, COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }
        
        // Apply CLAHE for better contrast
        Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(2.0);
        clahe->setTilesGridSize(Size(8, 8));
        clahe->apply(gray, gray);
        
        // Create directional kernels for vein detection
        vector<Mat> directional_kernels;
        
        // Horizontal kernel
        Mat kernel_h = (Mat_<float>(1, 9) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
        directional_kernels.push_back(kernel_h);
        
        // Vertical kernel
        Mat kernel_v = (Mat_<float>(9, 1) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
        directional_kernels.push_back(kernel_v);
        
        // Diagonal kernels (45 and 135 degrees)
        Mat kernel_d1 = (Mat_<float>(5, 5) << 
            -1, -1, 0, 1, 1,
            -1, -1, 0, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 0, -1, -1,
            1, 1, 0, -1, -1);
        directional_kernels.push_back(kernel_d1);
        
        Mat kernel_d2 = (Mat_<float>(5, 5) << 
            1, 1, 0, -1, -1,
            1, 1, 0, -1, -1,
            0, 0, 0, 0, 0,
            -1, -1, 0, 1, 1,
            -1, -1, 0, 1, 1);
        directional_kernels.push_back(kernel_d2);
        
        // Apply directional filters
        vector<Mat> directional_responses;
        for (const Mat& kernel : directional_kernels) {
            Mat response;
            filter2D(gray, response, CV_32F, kernel);
            Mat abs_response;
            convertScaleAbs(response, abs_response);
            directional_responses.push_back(abs_response);
        }
        
        // Combine directional responses
        Mat vein_enhanced = Mat::zeros(gray.size(), CV_8UC1);
        for (const Mat& response : directional_responses) {
            max(vein_enhanced, response, vein_enhanced);
        }
        
        // Combine with traditional edge detection
        Mat combined_edges = combineEdgeDetectors(gray);
        
        // Final combination
        Mat final_veins;
        addWeighted(vein_enhanced, 0.6, combined_edges, 0.4, 0, final_veins);
        
        return final_veins;
    }
    
    // Skeletonization algorithm for biological modeling
    Mat skeletonize(const Mat& binary_image) {
        Mat skeleton = Mat::zeros(binary_image.size(), CV_8UC1);
        Mat temp, eroded;
        Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
        
        Mat image = binary_image.clone();
        
        bool done = false;
        while (!done) {
            erode(image, eroded, element);
            dilate(eroded, temp, element);
            subtract(image, temp, temp);
            bitwise_or(skeleton, temp, skeleton);
            eroded.copyTo(image);
            
            done = (countNonZero(image) == 0);
        }
        
        return skeleton;
    }
    
    // Extract quantitative vein features
    struct VeinFeatures {
        double total_vein_length;
        double average_vein_width;
        int num_branch_points;
        int num_end_points;
        double vein_density;
        double tortuosity;
        vector<double> vein_angles;
        double fractal_dimension;
    };
    
    VeinFeatures extractQuantitativeFeatures(const Mat& skeleton_image, const Mat& original_veins) {
        VeinFeatures features;
        
        // Calculate total vein length (number of skeleton pixels)
        features.total_vein_length = countNonZero(skeleton_image);
        
        // Calculate vein density
        double total_area = skeleton_image.rows * skeleton_image.cols;
        features.vein_density = features.total_vein_length / total_area;
        
        // Find branch points and end points
        Mat kernel = (Mat_<uchar>(3,3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);
        Mat neighbor_count;
        filter2D(skeleton_image, neighbor_count, CV_8UC1, kernel);
        
        // Branch points have more than 2 neighbors
        Mat branch_points;
        threshold(neighbor_count, branch_points, 2*255, 255, THRESH_BINARY);
        bitwise_and(branch_points, skeleton_image, branch_points);
        features.num_branch_points = countNonZero(branch_points) / 255;
        
        // End points have exactly 1 neighbor
        Mat end_points, temp;
        threshold(neighbor_count, end_points, 1*255, 255, THRESH_BINARY);
        threshold(neighbor_count, temp, 2*255, 255, THRESH_BINARY_INV);
        bitwise_and(end_points, temp, end_points);
        bitwise_and(end_points, skeleton_image, end_points);
        features.num_end_points = countNonZero(end_points) / 255;
        
        // Calculate average vein width using distance transform
        Mat dist_transform;
        distanceTransform(original_veins, dist_transform, DIST_L2, 3);
        Scalar mean_dist = mean(dist_transform, original_veins);
        features.average_vein_width = mean_dist[0] * 2; // Diameter = 2 * radius
        
        // Calculate tortuosity (simplified as ratio of actual length to straight-line distance)
        vector<vector<Point>> contours;
        findContours(skeleton_image, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        
        double total_tortuosity = 0.0;
        int valid_contours = 0;
        
        for (const auto& contour : contours) {
            if (contour.size() > 10) { // Only consider significant contours
                double arc_length = arcLength(contour, false);
                double straight_distance = norm(contour.front() - contour.back());
                if (straight_distance > 0) {
                    total_tortuosity += arc_length / straight_distance;
                    valid_contours++;
                }
            }
        }
        
        features.tortuosity = (valid_contours > 0) ? total_tortuosity / valid_contours : 1.0;
        
        // Simplified fractal dimension calculation using box-counting method
        features.fractal_dimension = calculateFractalDimension(skeleton_image);
        
        return features;
    }
    
    // Simplified fractal dimension calculation
    double calculateFractalDimension(const Mat& binary_image) {
        vector<int> box_sizes = {2, 4, 8, 16, 32};
        vector<double> log_box_sizes, log_counts;
        
        for (int box_size : box_sizes) {
            int count = 0;
            for (int y = 0; y < binary_image.rows; y += box_size) {
                for (int x = 0; x < binary_image.cols; x += box_size) {
                    Rect box(x, y, min(box_size, binary_image.cols - x), 
                            min(box_size, binary_image.rows - y));
                    Mat roi = binary_image(box);
                    if (countNonZero(roi) > 0) {
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                log_box_sizes.push_back(log(1.0 / box_size));
                log_counts.push_back(log(count));
            }
        }
        
        // Calculate slope using linear regression
        if (log_box_sizes.size() < 2) return 1.0;
        
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        int n = log_box_sizes.size();
        
        for (int i = 0; i < n; i++) {
            sum_x += log_box_sizes[i];
            sum_y += log_counts[i];
            sum_xy += log_box_sizes[i] * log_counts[i];
            sum_x2 += log_box_sizes[i] * log_box_sizes[i];
        }
        
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        return slope;
    }
    
    // Main processing function
    pair<bool, VeinFeatures> processLeafImage(const Mat& input) {
        VeinFeatures empty_features = {};
        
        // Step 1: Check if image contains a leaf
        if (!isLeafImage(input)) {
            cout << "Image does not appear to contain a leaf." << endl;
            return make_pair(false, empty_features);
        }
        
        cout << "Leaf detected! Processing vein structure..." << endl;
        
        // Step 2: Detect veins using combined algorithm
        Mat vein_image = detectVeinsAllDirections(input);
        
        // Step 3: Post-process vein image
        Mat binary_veins;
        threshold(vein_image, binary_veins, 0, 255, THRESH_BINARY + THRESH_OTSU);
        
        // Step 4: Apply morphological operations to clean up
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(binary_veins, binary_veins, MORPH_CLOSE, kernel);
        morphologyEx(binary_veins, binary_veins, MORPH_OPEN, kernel);
        
        // Step 5: Skeletonize for biological modeling
        Mat skeleton = skeletonize(binary_veins);
        
        // Step 6: Extract quantitative features
        VeinFeatures features = extractQuantitativeFeatures(skeleton, binary_veins);
        
        // Display results
        displayResults(input, vein_image, binary_veins, skeleton, features);
        
        return make_pair(true, features);
    }
    
    // Display processing results
    void displayResults(const Mat& original, const Mat& veins, const Mat& binary, 
                       const Mat& skeleton, const VeinFeatures& features) {
        // Create a combined display image
        Mat display;
        
        // Convert single channel images to 3-channel for display
        Mat veins_color, binary_color, skeleton_color;
        cvtColor(veins, veins_color, COLOR_GRAY2BGR);
        cvtColor(binary, binary_color, COLOR_GRAY2BGR);
        cvtColor(skeleton, skeleton_color, COLOR_GRAY2BGR);
        
        // Make skeleton more visible (red)
        skeleton_color.setTo(Scalar(0, 0, 255), skeleton);
        
        // Combine images in a 2x2 grid
        Mat top_row, bottom_row;
        hconcat(original, veins_color, top_row);
        hconcat(binary_color, skeleton_color, bottom_row);
        vconcat(top_row, bottom_row, display);
        
        // Add text labels
        putText(display, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(display, "Vein Detection", Point(original.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(display, "Binary Veins", Point(10, original.rows + 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(display, "Skeleton", Point(original.cols + 10, original.rows + 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        
        // Print quantitative features
        cout << "\n=== QUANTITATIVE VEIN FEATURES ===" << endl;
        cout << "Total Vein Length: " << features.total_vein_length << " pixels" << endl;
        cout << "Average Vein Width: " << features.average_vein_width << " pixels" << endl;
        cout << "Number of Branch Points: " << features.num_branch_points << endl;
        cout << "Number of End Points: " << features.num_end_points << endl;
        cout << "Vein Density: " << features.vein_density << endl;
        cout << "Average Tortuosity: " << features.tortuosity << endl;
        cout << "Fractal Dimension: " << features.fractal_dimension << endl;
        
        imshow("Leaf Vein Analysis Results", display);
        waitKey(0);
    }
    
    // Species detection placeholder (would require trained classifier)
    string detectSpecies(const Mat& input, const VeinFeatures& features) {
        // This would typically involve:
        // 1. Extract additional morphological features (leaf shape, size, etc.)
        // 2. Use the vein features as input to a trained classifier
        // 3. Return the predicted species name
        
        // Placeholder implementation based on simple rules
        if (features.vein_density > 0.1 && features.num_branch_points > 50) {
            return "Complex venation - possibly dicot species";
        } else if (features.tortuosity > 1.5) {
            return "Highly curved veins - possibly monocot species";
        } else {
            return "Simple venation pattern - species unknown";
        }
    }
};

// Main function demonstrating usage
int main() {
    AdvancedLeafVeinDetector detector;
    
    // Load an image (replace with actual image path)
    string image_path = "leaf_sample.jpg";
    Mat input = imread(image_path);
    
    if (input.empty()) {
        cout << "Error: Could not load image from " << image_path << endl;
        cout << "Please ensure the image file exists and the path is correct." << endl;
        return -1;
    }
    
    // Process the leaf image
    auto result = detector.processLeafImage(input);
    
    if (result.first) {
        // If leaf was detected, attempt species identification
        string species = detector.detectSpecies(input, result.second);
        cout << "\nPredicted classification: " << species << endl;
    }
    
    return 0;
}

// Compile with:
// g++ -std=c++17 -O3 algorithm.cpp -o leaf_detector -I"C:\opencv\opencv\build\include" -L"C:\opencv\opencv\build\x64\vc16\lib" -lopencv_core460 -lopencv_imgproc460 -lopencv_highgui460 -lopencv_imgcodecs460

