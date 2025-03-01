#include<stdio.h>
#include<omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include<time.h>
#include<stdlib.h>
#include <windows.h>
using namespace std;

#define MAX_THREADS 64       // 最大线程数
#define SUBDATANUM 2000000   // 每个线程处理的数据量
#define DATANUM (SUBDATANUM * MAX_THREADS)   // 总数据量

// 求和和最大值时使用的线程数
#define THREADS_NUM 64

// 待测试数据
float rawFloatData[DATANUM]; // 原始浮点数据数组
// 求最大值时的中间数据
float a[DATANUM];

//求和函数
float mysum(const float data[], const int len) {
    double sum = 0.0; // 中间变量，用于累加总和
    for (size_t i = 0; i < len; i++) {
       // sum += log(sqrt(data[i]));
        sum += data[i];

    }
    return sum;
}

//最大值函数

float mymax(const float data[], const int len) {
  
    // 寻找最大值
    float max = 0.0f;
    for (size_t i = 0; i < DATANUM; i++) {
        if (data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

//交换函数
template<typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

//堆化
void heapify(float* arr, int idx, int heap_size) {
    int largest = idx; // 假设当前节点为最大节点
    int left_child = 2 * idx + 1;  // 左子节点索引
    int right_child = 2 * idx + 2; // 右子节点索引

    // 比较左子节点
    if (left_child < heap_size && arr[left_child] > arr[largest])
        largest = left_child;

    // 比较右子节点
    if (right_child < heap_size && arr[right_child] > arr[largest])
        largest = right_child;

    // 如果最大值不是当前节点，交换并递归调整
    if (largest != idx) {
        mySwap(arr[idx], arr[largest]);
        heapify(arr, largest, heap_size);
    }
}

//堆排
void heapSort(float* arr, int len) {
    // 构建最大堆
    for (int i = len / 2 - 1; i >= 0; i--)
        heapify(arr, i, len);

    // 逐步将最大元素移到末尾，并调整堆
    for (int i = len - 1; i >= 0; i--) {
        mySwap(arr[0], arr[i]);
        heapify(arr, 0, i);
    }
}

//检查数组是否有序
 
bool checkSort(const float data[], const int len) {
    for (int i = 0; i < len - 1; i++) {
        if (data[i + 1] < data[i])
            return false;
    }
    return true;
}

//打印数组中的部分元素（调试用）

void printArray(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << endl;
    }
    cout << endl;
}

int main() {
    // 数据初始化
    for (size_t i = 0; i < DATANUM; i++) {
        //rawFloatData[i] = float(i + 1);
        rawFloatData[i] = log(sqrt(float(i + 1)));

    }

    // 计时器初始化
    LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3, f;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&time_start);

    // 求和
    float SumResult = mysum(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_1);

    // 求最大值
    float MaxResult = mymax(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_2);

    // 堆排序
   heapSort(rawFloatData, DATANUM);
   QueryPerformanceCounter(&time_over_3);

    // 输出结果
    cout << "总时间:" << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    cout << "总和:" << SumResult << endl;
    cout << "最大值:" << MaxResult << endl;
    cout << "求和时间:" << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    cout << "最大值时间:" << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << endl;
    cout << "排序时间:" << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << endl;

    // 检查排序
    bool checkflag = checkSort(rawFloatData, 2000000);
    printf("检查排序：%d", checkflag);
    cout << endl;
}
