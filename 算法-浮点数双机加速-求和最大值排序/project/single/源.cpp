#include<stdio.h>
#include<omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include<time.h>
#include<stdlib.h>
#include <windows.h>
using namespace std;

#define MAX_THREADS 64       // ����߳���
#define SUBDATANUM 2000000   // ÿ���̴߳����������
#define DATANUM (SUBDATANUM * MAX_THREADS)   // ��������

// ��ͺ����ֵʱʹ�õ��߳���
#define THREADS_NUM 64

// ����������
float rawFloatData[DATANUM]; // ԭʼ������������
// �����ֵʱ���м�����
float a[DATANUM];

//��ͺ���
float mysum(const float data[], const int len) {
    double sum = 0.0; // �м�����������ۼ��ܺ�
    for (size_t i = 0; i < len; i++) {
       // sum += log(sqrt(data[i]));
        sum += data[i];

    }
    return sum;
}

//���ֵ����

float mymax(const float data[], const int len) {
  
    // Ѱ�����ֵ
    float max = 0.0f;
    for (size_t i = 0; i < DATANUM; i++) {
        if (data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

//��������
template<typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

//�ѻ�
void heapify(float* arr, int idx, int heap_size) {
    int largest = idx; // ���赱ǰ�ڵ�Ϊ���ڵ�
    int left_child = 2 * idx + 1;  // ���ӽڵ�����
    int right_child = 2 * idx + 2; // ���ӽڵ�����

    // �Ƚ����ӽڵ�
    if (left_child < heap_size && arr[left_child] > arr[largest])
        largest = left_child;

    // �Ƚ����ӽڵ�
    if (right_child < heap_size && arr[right_child] > arr[largest])
        largest = right_child;

    // ������ֵ���ǵ�ǰ�ڵ㣬�������ݹ����
    if (largest != idx) {
        mySwap(arr[idx], arr[largest]);
        heapify(arr, largest, heap_size);
    }
}

//����
void heapSort(float* arr, int len) {
    // ��������
    for (int i = len / 2 - 1; i >= 0; i--)
        heapify(arr, i, len);

    // �𲽽����Ԫ���Ƶ�ĩβ����������
    for (int i = len - 1; i >= 0; i--) {
        mySwap(arr[0], arr[i]);
        heapify(arr, 0, i);
    }
}

//��������Ƿ�����
 
bool checkSort(const float data[], const int len) {
    for (int i = 0; i < len - 1; i++) {
        if (data[i + 1] < data[i])
            return false;
    }
    return true;
}

//��ӡ�����еĲ���Ԫ�أ������ã�

void printArray(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << endl;
    }
    cout << endl;
}

int main() {
    // ���ݳ�ʼ��
    for (size_t i = 0; i < DATANUM; i++) {
        //rawFloatData[i] = float(i + 1);
        rawFloatData[i] = log(sqrt(float(i + 1)));

    }

    // ��ʱ����ʼ��
    LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3, f;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&time_start);

    // ���
    float SumResult = mysum(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_1);

    // �����ֵ
    float MaxResult = mymax(rawFloatData, DATANUM);
    QueryPerformanceCounter(&time_over_2);

    // ������
   heapSort(rawFloatData, DATANUM);
   QueryPerformanceCounter(&time_over_3);

    // ������
    cout << "��ʱ��:" << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    cout << "�ܺ�:" << SumResult << endl;
    cout << "���ֵ:" << MaxResult << endl;
    cout << "���ʱ��:" << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    cout << "���ֵʱ��:" << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << endl;
    cout << "����ʱ��:" << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << endl;

    // �������
    bool checkflag = checkSort(rawFloatData, 2000000);
    printf("�������%d", checkflag);
    cout << endl;
}
