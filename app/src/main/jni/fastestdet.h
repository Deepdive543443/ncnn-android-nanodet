//
// Created by swt on 04/04/2024.
//

#ifndef FASTESTDET_H
#define FASTESTDET_H

#include <opencv2/core/core.hpp>
#include <net.h>

static inline float fast_exp(float x)
{
    union {
        __uint32_t i;
        float      f;
    } v;
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
#define FAST_SIGMOID(X) (1.0f / (1.0f + fast_exp(-X)))
#define FAST_TANH(X)    (2.f / (1.f + fast_exp(-2 * X)) - 1.f)

typedef struct Object {
    cv::Rect_<float> rect;
    int              label;
    float            prob;
} Object;

class FastestDet {
   public:
    FastestDet();

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals,
             bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.65f,
               float nms_threshold = 0.65f);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

   private:
    ncnn::Net                   fastestdet;
    int                         target_size;
    float                       mean_vals[3];
    float                       norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator         workspace_pool_allocator;
};

#endif  // FASTESTDET_H
