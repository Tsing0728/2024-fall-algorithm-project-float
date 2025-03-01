#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <windows.h>
#include <string>
#include <sstream>
#include <thread>       // 多线程支持（可能未必需要）
#include <immintrin.h>  // AVX 指令集支持
#include <float.h>      // 用于 FLT_MAX
#include <cfloat>
using namespace std;

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)

// Sum & Max
#define THREADS_NUM 64

//sum & max
#define THREADS_NUM 64
#define ALIGNMENT 32   // AVX 使用 256 位寄存器，内存对齐 32 字节

alignas(32) float rawFloatData[DATANUM];



// 求和函数 (OpenMP 加速+AVX)
float sumOpenMp_AVX(const float data[], const int len) {
    __m256 sum_vec = _mm256_setzero_ps();  // 初始化 AVX 累加器为零
    float sum = 0.0f;

#pragma omp parallel num_threads(THREADS_NUM)
    {
        __m256 local_sum = _mm256_setzero_ps();  // 每个线程的局部累加器
        int thread_id = omp_get_thread_num();
        int chunk_size = len / THREADS_NUM;
        int start = thread_id * chunk_size;
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size;

        for (int i = start; i <= end - 8; i += 8) {
            __m256 vec = _mm256_load_ps(&data[i]);  // 使用对齐加载
            local_sum = _mm256_add_ps(local_sum, vec);
        }

        // 将局部 AVX 累加器存储到临时数组
        float temp[8] alignas(32);
        _mm256_store_ps(temp, local_sum);
        float local_result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // 处理剩余部分
        for (int i = end - (end % 8); i < end; i++) {
            local_result += data[i];
        }

#pragma omp critical
        {
            sum += local_result;
        }
    }

    return sum;
}



 // 求最大值函数 (OpenMP 加速+AVX)
float maxOpenMp_AVX(const float data[], const int len) {
    float max_values[THREADS_NUM] = { -FLT_MAX };  // 存储每个线程的最大值,初始值设置为 -FLT_MAX，一个极小的浮点数，保证任何输入数据都会比它大。
    int chunk_size = len / THREADS_NUM;  // 每个线程处理的数据量

#pragma omp parallel num_threads(THREADS_NUM)
    {
        int thread_id = omp_get_thread_num();  // 获取当前线程 ID
        int start = thread_id * chunk_size;    // 每个线程的起始索引
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size; // 最后一个线程处理剩余部分

        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);  // 初始化 AVX 最大值向量


        // 使用 AVX 并行处理 8 个浮点数一组
        int i = start;
        for (; i <= end - 8; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]); // 加载 8 个 float 数据
            max_vec = _mm256_max_ps(max_vec, vec);  // 比较 8 个 float 的最大值
        }

        // 将 AVX 寄存器内的 8 个最大值提取到数组
        float temp[8];
        _mm256_storeu_ps(temp, max_vec);

        // 在本地线程中比较 AVX 结果
        float local_max = temp[0];
        for (int j = 1; j < 8; j++) {
            if (temp[j] > local_max) {
                local_max = temp[j];
            }
        }

        // 处理剩余的元素
        for (; i < end; i++) {
            if (data[i] > local_max) {
                local_max = data[i];
            }
        }

        // 保存每个线程的最大值
        max_values[thread_id] = local_max;
    }

    // 主线程合并各线程的最大值
    float global_max = max_values[0];
    for (int i = 1; i < THREADS_NUM; i++) {
        if (max_values[i] > global_max) {
            global_max = max_values[i];
        }
    }

    return global_max;
}



// 自定义交换函数
template<typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// 堆化函数
void heapify(float* arr, int idx, int heap_size) {
    int largest = idx;
    int left_child = 2 * idx + 1;
    int right_child = 2 * idx + 2;

    if (left_child < heap_size && arr[left_child] > arr[largest])
        largest = left_child;
    if (right_child < heap_size && arr[right_child] > arr[largest])
        largest = right_child;

    if (largest != idx) {
        mySwap(arr[idx], arr[largest]);
        heapify(arr, largest, heap_size);
    }
}

// 堆排序 (OpenMP 加速堆构建部分)
void heapSort(float* arr, int len) {
    // 1. 构建最大堆
//#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = len / 2 - 1; i >= 0; i--)
        heapify(arr, i, len);

    // 2. 提取最大值并调整堆
    for (int i = len - 1; i >= 0; i--) {
        mySwap(arr[0], arr[i]);
        heapify(arr, 0, i); // 调整剩余堆
    }
}




// 检查排序是否成功
bool checkSort(const float data[], const int len) {
    bool flag = 1;
    for (int i = 0; i < len - 1; i++) {
        flag = data[i + 1] >= data[i];
        if (!flag)
            break;
    }
    return flag;
}

// 打印部分数组 (调试用)
void printArray(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main() {
    // 数据初始化
    for (size_t i = 0; i < DATANUM; i++) {
        rawFloatData[i] = log(sqrt(float(i + 1)));
    }

    _LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3;
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&time_start);

    // 求和
    float SumResult = 0.0f;
    SumResult = sumOpenMp_AVX(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_1);

    // 求最大值
    float MaxResult = 0.0f;
    MaxResult = maxOpenMp_AVX(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_2);

    // 堆排序
   // parallelHeapSort(rawFloatData, DATANUM);
    heapSort(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_3);

    // 输出结果
    std::cout << "Time Consumed: " << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << "s" << endl;
    cout << "Sum: " << SumResult << endl;
    cout << "Max: " << MaxResult << endl;
    std::cout << "Sum Time Consumed: " << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << "s" << endl;
    std::cout << "Max Time Consumed: " << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << "s" << endl;
    std::cout << "Sort Time Consumed: " << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << "s" << endl;

    // 检查排序
    bool checkflag = checkSort(rawFloatData, DATANUM);
    printf("检查排序：%d\n", checkflag);

    //printArray(rawFloatData, 30);

    return 0;
}
