// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "fastestdet.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpu.h"

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.width * objects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

FastestDet::FastestDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int FastestDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    fastestdet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    fastestdet.opt = ncnn::Option();

#if NCNN_VULKAN
    fastestdet.opt.use_vulkan_compute = use_gpu;
#endif

    fastestdet.opt.num_threads = ncnn::get_big_cpu_count();
    fastestdet.opt.blob_allocator = &blob_pool_allocator;
    fastestdet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "FastestDet.param", modeltype);
    sprintf(modelpath, "FastestDet.bin", modeltype);

    fastestdet.load_param(parampath);
    fastestdet.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int FastestDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    fastestdet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodet_plus.opt = ncnn::Option();

#if NCNN_VULKAN
    fastestdet.opt.use_vulkan_compute = use_gpu;
#endif
    fastestdet.opt.use_winograd_convolution = true;
    fastestdet.opt.use_sgemm_convolution = true;
    fastestdet.opt.use_int8_inference = true;

    fastestdet.opt.num_threads = ncnn::get_big_cpu_count();
    fastestdet.opt.blob_allocator = &blob_pool_allocator;
    fastestdet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "FastestDet.param", modeltype);
    sprintf(modelpath, "FastestDet.bin", modeltype);

    fastestdet.load_param(mgr, parampath);
    fastestdet.load_model(mgr, modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int FastestDet::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    ncnn::Mat out;
    {
        ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR, rgb.cols,
                                                        rgb.rows, target_size, target_size);

        input.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = detector.create_extractor();

        ex.input("data", input);
        ex.extract("output", out);
    }

    int                  c_step = out.cstep;
    float                obj_score;
    std::vector<Object> proposals;

    int count = 0;

    for (int h = 0; h < out.h; h++) {
        float *ptr = out.row(h);
        for (int w = 0; w < out.w; w++) {
            float obj_score     = ptr[0];
            float max_cls_score = 0.0;
            int   max_cls_idx   = -1;

            for (int c = 0; c < num_class; c++) {
                float cls_score = ptr[(c + 5) * c_step];
                if (cls_score > max_cls_score) {
                    max_cls_score = cls_score;
                    max_cls_idx   = c;
                }
            }

            if (pow(max_cls_score, 0.4) * pow(obj_score, 0.6) > 0.65) {
                float x_offset   = FAST_TANH(ptr[c_step]);
                float y_offset   = FAST_TANH(ptr[c_step * 2]);
                float box_width  = FAST_SIGMOID(ptr[c_step * 3]);
                float box_height = FAST_SIGMOID(ptr[c_step * 4]);
                float x_center   = (w + x_offset) / out.w;
                float y_center   = (h + y_offset) / out.h;

                Object obj;
                obj.rect.x = (x_center - 0.5 * box_width) * rgb.cols;
                obj.rect.y = (y_center - 0.5 * box_height) * rgb.rows;
                obj.rect.width =  box_width * rgb.cols;
                obj.rect.height = box_height * rgb.rows;
                obj.label = max_cls_idx;
                obj.prob = obj_score;
//                info.x1    = (x_center - 0.5 * box_width) * ocv_input.cols;
//                info.y1    = (y_center - 0.5 * box_height) * ocv_input.rows;
//                info.x2    = (x_center + 0.5 * box_width) * ocv_input.cols;
//                info.y2    = (y_center + 0.5 * box_height) * ocv_input.rows;
//                info.label = max_cls_idx;
//                info.score = obj_score;

                results.push_back(info);
            }
            ptr++;
        }
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // Transfer to x0y0x1y1
        float x0 = objects[i].rect.x;
        float y0 = objects[i].rect.y;
        float x1 = objects[i].rect.x + objects[i].rect.width;
        float y1 = objects[i].rect.y + objects[i].rect.height;

        // clip
        x0 = std::max(std::min(x0, (float)(rgb.cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(rgb.rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(rgb.cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(rgb.rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    return 1;
}

int FastestDet::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };

    static const unsigned char colors[19][3] = {
            { 54,  67, 244},
            { 99,  30, 233},
            {176,  39, 156},
            {183,  58, 103},
            {181,  81,  63},
            {243, 150,  33},
            {244, 169,   3},
            {212, 188,   0},
            {136, 150,   0},
            { 80, 175,  76},
            { 74, 195, 139},
            { 57, 220, 205},
            { 59, 235, 255},
            {  7, 193, 255},
            {  0, 152, 255},
            { 34,  87, 255},
            { 72,  85, 121},
            {158, 158, 158},
            {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }
    return 0;
}
