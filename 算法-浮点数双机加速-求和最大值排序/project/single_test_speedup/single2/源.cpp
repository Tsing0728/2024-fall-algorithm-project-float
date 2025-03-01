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
#include <thread>       // ���߳�֧�֣�����δ����Ҫ��
#include <immintrin.h>  // AVX ָ�֧��
#include <float.h>      // ���� FLT_MAX
#include <cfloat>
using namespace std;

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)

// Sum & Max
#define THREADS_NUM 64

//sum & max
#define THREADS_NUM 64
#define ALIGNMENT 32   // AVX ʹ�� 256 λ�Ĵ������ڴ���� 32 �ֽ�

alignas(32) float rawFloatData[DATANUM];



// ��ͺ��� (OpenMP ����+AVX)
float sumOpenMp_AVX(const float data[], const int len) {
    __m256 sum_vec = _mm256_setzero_ps();  // ��ʼ�� AVX �ۼ���Ϊ��
    float sum = 0.0f;

#pragma omp parallel num_threads(THREADS_NUM)
    {
        __m256 local_sum = _mm256_setzero_ps();  // ÿ���̵߳ľֲ��ۼ���
        int thread_id = omp_get_thread_num();
        int chunk_size = len / THREADS_NUM;
        int start = thread_id * chunk_size;
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size;

        for (int i = start; i <= end - 8; i += 8) {
            __m256 vec = _mm256_load_ps(&data[i]);  // ʹ�ö������
            local_sum = _mm256_add_ps(local_sum, vec);
        }

        // ���ֲ� AVX �ۼ����洢����ʱ����
        float temp[8] alignas(32);
        _mm256_store_ps(temp, local_sum);
        float local_result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // ����ʣ�ಿ��
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



 // �����ֵ���� (OpenMP ����+AVX)
float maxOpenMp_AVX(const float data[], const int len) {
    float max_values[THREADS_NUM] = { -FLT_MAX };  // �洢ÿ���̵߳����ֵ,��ʼֵ����Ϊ -FLT_MAX��һ����С�ĸ���������֤�κ��������ݶ��������
    int chunk_size = len / THREADS_NUM;  // ÿ���̴߳����������

#pragma omp parallel num_threads(THREADS_NUM)
    {
        int thread_id = omp_get_thread_num();  // ��ȡ��ǰ�߳� ID
        int start = thread_id * chunk_size;    // ÿ���̵߳���ʼ����
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size; // ���һ���̴߳���ʣ�ಿ��

        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);  // ��ʼ�� AVX ���ֵ����


        // ʹ�� AVX ���д��� 8 ��������һ��
        int i = start;
        for (; i <= end - 8; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]); // ���� 8 �� float ����
            max_vec = _mm256_max_ps(max_vec, vec);  // �Ƚ� 8 �� float �����ֵ
        }

        // �� AVX �Ĵ����ڵ� 8 �����ֵ��ȡ������
        float temp[8];
        _mm256_storeu_ps(temp, max_vec);

        // �ڱ����߳��бȽ� AVX ���
        float local_max = temp[0];
        for (int j = 1; j < 8; j++) {
            if (temp[j] > local_max) {
                local_max = temp[j];
            }
        }

        // ����ʣ���Ԫ��
        for (; i < end; i++) {
            if (data[i] > local_max) {
                local_max = data[i];
            }
        }

        // ����ÿ���̵߳����ֵ
        max_values[thread_id] = local_max;
    }

    // ���̺߳ϲ����̵߳����ֵ
    float global_max = max_values[0];
    for (int i = 1; i < THREADS_NUM; i++) {
        if (max_values[i] > global_max) {
            global_max = max_values[i];
        }
    }

    return global_max;
}



// �Զ��彻������
template<typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// �ѻ�����
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

// ������ (OpenMP ���ٶѹ�������)
void heapSort(float* arr, int len) {
    // 1. ��������
//#pragma omp parallel for num_threads(THREADS_NUM)
    for (int i = len / 2 - 1; i >= 0; i--)
        heapify(arr, i, len);

    // 2. ��ȡ���ֵ��������
    for (int i = len - 1; i >= 0; i--) {
        mySwap(arr[0], arr[i]);
        heapify(arr, 0, i); // ����ʣ���
    }
}




// ��������Ƿ�ɹ�
bool checkSort(const float data[], const int len) {
    bool flag = 1;
    for (int i = 0; i < len - 1; i++) {
        flag = data[i + 1] >= data[i];
        if (!flag)
            break;
    }
    return flag;
}

// ��ӡ�������� (������)
void printArray(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main() {
    // ���ݳ�ʼ��
    for (size_t i = 0; i < DATANUM; i++) {
        rawFloatData[i] = log(sqrt(float(i + 1)));
    }

    _LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3;
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&time_start);

    // ���
    float SumResult = 0.0f;
    SumResult = sumOpenMp_AVX(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_1);

    // �����ֵ
    float MaxResult = 0.0f;
    MaxResult = maxOpenMp_AVX(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_2);

    // ������
   // parallelHeapSort(rawFloatData, DATANUM);
    heapSort(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_3);

    // ������
    std::cout << "Time Consumed: " << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << "s" << endl;
    cout << "Sum: " << SumResult << endl;
    cout << "Max: " << MaxResult << endl;
    std::cout << "Sum Time Consumed: " << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << "s" << endl;
    std::cout << "Max Time Consumed: " << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << "s" << endl;
    std::cout << "Sort Time Consumed: " << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << "s" << endl;

    // �������
    bool checkflag = checkSort(rawFloatData, DATANUM);
    printf("�������%d\n", checkflag);

    //printArray(rawFloatData, 30);

    return 0;
}
