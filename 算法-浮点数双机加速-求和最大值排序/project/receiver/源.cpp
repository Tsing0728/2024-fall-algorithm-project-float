#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <omp.h>        // OpenMP ֧��
#include <algorithm>
#include <thread>       // ���߳�֧�֣�����δ����Ҫ��
#include <time.h>
#include <stdlib.h>
#include <windows.h>    // Windows API ֧�� (���� QueryPerformanceCounter)
#include <immintrin.h>  // AVX ָ�֧��
#include <float.h>      // ���� FLT_MAX
#include <cfloat>

using namespace std;

#pragma comment(lib, "ws2_32.lib")

#define MAX_THREADS 64
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*�����ֵ����������*/

//sum & max
#define THREADS_NUM 64


#define ALIGNMENT 32   // AVX ʹ�� 256 λ�Ĵ������ڴ���� 32 �ֽ�

alignas(32) float rawFloatData[DATANUM];
alignas(32) char rawCharArray[DATANUM * 4];
alignas(32) float max_values[THREADS_NUM];


void sendMessage(std::string message, SOCKET clientSocket) {
    if (send(clientSocket, message.c_str(), message.length(), 0) == SOCKET_ERROR) {
        std::cerr << "Can't send data, Err #" << WSAGetLastError() << std::endl;
    }
    char buffer[4096];
    int bytesReceived;
    memset(buffer, 0, sizeof buffer);
    bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    if (bytesReceived == 0) {
        std::cerr << "Connection closed by server" << std::endl;
    }
    else if (bytesReceived == SOCKET_ERROR) {
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;
    }
    std::cout << "Received: " << buffer << std::endl;
}

// ��ͺ��� (OpenMP ����)
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
        float local_result = temp[0] + temp[1] + temp[2] + temp[3] +
            temp[4] + temp[5] + temp[6] + temp[7];

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


float maxOpenMp(const float data[], const int len) {
    float max_values[THREADS_NUM] = { 0 };  // �洢ÿ���̵߳����ֵ
    int chunk_size = len / THREADS_NUM;  // ÿ���̴߳����������

#pragma omp parallel num_threads(THREADS_NUM)
    {
        int thread_id = omp_get_thread_num();  // ��ȡ��ǰ�߳� ID
        int start = thread_id * chunk_size;    // ÿ���̵߳���ʼ����
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size; // ���һ���̴߳���ʣ�ಿ��

        float local_max = data[start];
        for (int i = start + 1; i < end; i++) {
            if (data[i] > local_max) {
                local_max = data[i];
            }
        }
        max_values[thread_id] = local_max;  // �����ֲ߳̾����ֵ
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



float maxOpenMp_AVX(const float data[], const int len) {
    float max_values[THREADS_NUM] = { -FLT_MAX };  // �洢ÿ���̵߳����ֵ
    int chunk_size = len / THREADS_NUM;  // ÿ���̴߳����������

#pragma omp parallel num_threads(THREADS_NUM)
    {
        int thread_id = omp_get_thread_num();  // ��ȡ��ǰ�߳� ID
        int start = thread_id * chunk_size;    // ÿ���̵߳���ʼ����
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size; // ���һ���̴߳���ʣ�ಿ��

        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);  // ��ʼ�� AVX ���ֵ����

        // ʹ�� AVX ���� 8 ��������һ��
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

// �Զ��彻�����������ڽ�������������ֵ
// ģ��ʹ��֧�ֲ�ͬ���͵����ݽ���
template<typename T>
void mySwap(T& a, T& b) {
    T temp = a; // �� a ��ֵ�洢����ʱ���� temp ��
    a = b;      // �� b ��ֵ���� a
    b = temp;   // �� temp��ԭ a ��ֵ������ b
}

// �ѻ�����
void heapify_AVX(float* arr, int idx, int heap_size) {
    while (true) {
        int largest = idx;
        int left_child = 2 * idx + 1;
        int right_child = 2 * idx + 2;

        if (left_child < heap_size && arr[left_child] > arr[largest]) {
            largest = left_child;
        }
        if (right_child < heap_size && arr[right_child] > arr[largest]) {
            largest = right_child;
        }

        if (largest != idx) {
            mySwap(arr[idx], arr[largest]);
            idx = largest;  // �����³�
        }
        else {
            break;  // ���û�н���������
        }
    }
}


void heapSort_AVX(float* arr, int len) {
    // ��������
    for (int i = len / 2 - 1; i >= 0; i--) {
        heapify_AVX(arr, i, len);
    }

    // ��ȡԪ��
    for (int i = len - 1; i > 0; i--) {
        mySwap(arr[0], arr[i]);
        heapify_AVX(arr, 0, i);
    }
}

//�������
bool checkSort(const float data[], const int len)
{
    bool flag = 1;
    for (int i = 0; i < len - 1; i++) {
        flag = data[i + 1] >= data[i];
        if (flag == 0)
            break;
    }
    return flag;
}

int main() {
    // ��ʼ��WinSock
    WSADATA wsaData;
    WORD version = MAKEWORD(2, 2); 
    int wsResult = WSAStartup(version, &wsaData);  // ��ʼ��WinSock��
    if (wsResult != 0) {  // ����ʼ���Ƿ�ɹ�
        std::cerr << "Can't start WinSock, Err #" << wsResult << std::endl;  // ���������Ϣ
        return -1;  // �����ʼ��ʧ�ܣ��˳�����
    }

    // ����������Socket��ʹ��IPv4��TCPЭ��
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);  // ����TCP������Socket

    
    sockaddr_in hint;
    hint.sin_family = AF_INET;  
    hint.sin_port = htons(8899);  // ���ö˿ں�8899�����˿ں�ת��Ϊ�����ֽ�˳��
    hint.sin_addr.S_un.S_addr = INADDR_ANY;  // ����IP��ַΪINADDR_ANY����ʾ�����κοͻ��˵�����

    // �󶨷�����Socket��ָ����IP�Ͷ˿�
    bind(serverSocket, (sockaddr*)&hint, sizeof(hint));  // ��������Socket�󶨵�ָ����ַ�Ͷ˿�

    listen(serverSocket, 5);  // ���ü��������ͬʱ����5����������

    // ���ܿͻ�������
    sockaddr_in client;
    int clientSize = sizeof(client);  // �ͻ��˵�ַ�Ĵ�С
    SOCKET clientSocket = accept(serverSocket, (sockaddr*)&client, &clientSize);  // ���ܿͻ�������

    // ���տͻ��˷���������
    char buffer[4096];  // ���建�������ڽ�������
    int bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);  // �ӿͻ��˽�������
    if (bytesReceived == 0) {  // ������յ������ݳ���Ϊ0����ʾ�ͻ��˹ر�������
        std::cerr << "Connection closed by Client" << std::endl;  // ���������Ϣ
        return -1;  // �˳�����
    }
    else if (bytesReceived == SOCKET_ERROR) {  // ����������ݳ���
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;  // ���������Ϣ
        return -1;
    }

    // ������յ�������
    std::cout << "Received: " << buffer << std::endl;

    // ��ͻ��˷�����Ӧ��Ϣ
    send(clientSocket, "Received Data!", 14, 0);  // ����"Received Data!"���ͻ���

    // ������յ������ݣ������ʼֵ
    int start = atoi(buffer);  // �����յ����ַ���תΪ����

    for (size_t i = 0; i < DATANUM; i++) {  // ���ݳ�ʼ�����Ӻ�64000000��ʼ
        rawFloatData[i] = log(sqrt(float(start + i + 1)));  // ���㲢���rawFloatData����
    }

    // ��ʼ�����ܼ�ʱ��
    _LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3;  // �����ʱ����
    LARGE_INTEGER f;  // ��ʱ��Ƶ��
    QueryPerformanceFrequency(&f);  // ��ȡ��ʱ����Ƶ��
    QueryPerformanceCounter(&time_start);  // ��ʼ��ʱ

    // ����speedup��ͺ���
    float SumResult = 0.0f;
    SumResult = sumOpenMp_AVX(rawFloatData, DATANUM);  
    QueryPerformanceCounter(&time_over_1);  // ��¼��ͽ���ʱ��

    // ����speedup�����ֵ����
    float MaxResult = 0.0f;
    MaxResult = maxOpenMp_AVX(rawFloatData, DATANUM); 
    QueryPerformanceCounter(&time_over_2);  // ��¼���ֵ�������ʱ��

    // �����ݽ��ж�����
    heapSort_AVX(rawFloatData, DATANUM); 
    QueryPerformanceCounter(&time_over_3);  // ��¼�������ʱ��

    // ������ܲ��Խ��
    std::cout << "Time Consumed:" << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << std::endl;
    std::cout << SumResult << std::endl;  // �����ͽ��
    std::cout << MaxResult << std::endl;  // ������ֵ���
    std::cout << "Sum Time Consumed:" << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << std::endl;
    std::cout << "Max Time Consumed:" << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << std::endl;
    std::cout << "Sort Time Consumed:" << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << std::endl;

    // ��������Ƿ���ȷ
    bool checkflag;
    checkflag = checkSort(rawFloatData, 1000000);  // ����������Ƿ���ȷ
    printf("�������%d", checkflag);  // �����������
    std::cout << std::endl;

    // �����������װΪ�ַ��������͸��ͻ���
    string dataString = std::to_string(SumResult) + "|" + std::to_string(MaxResult) + "|" + std::to_string(checkflag);
    send(clientSocket, dataString.c_str(), dataString.size(), 0);  // ���ʹ��������ͻ���



    // ���տͻ��˵ķ�������
    memset(buffer, 0, sizeof buffer);  // ��ջ�����

    bytesReceived = recv(clientSocket, buffer, 4096, 0);  // ��������
    if (bytesReceived == 0) {  // ������յ������ݳ���Ϊ0����ʾ�ͻ��˹ر�������
        std::cerr << "Connection closed by server" << std::endl;
    }
    else if (bytesReceived == SOCKET_ERROR) {  // ����������ݳ���
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;
    }
    std::cout << "Received: " << buffer << std::endl;



    // ��������������ת��Ϊ�ַ�������з���
    memcpy(rawCharArray, rawFloatData, DATANUM * 4);  // ������������ת��Ϊ�ַ�����

    // ����rawFloatData����
    int sentLength = 0;  // �ѷ��͵��ֽ���
    int charLength = DATANUM * 4;  //float ����ռ 4 �ֽڣ����������ֽ���
    while (sentLength < charLength) {  // ѭ����������
        int toSend = charLength - sentLength;  // ���㻹�跢�͵��ֽ���
        if (toSend > 4096) {  // ÿ�η������4096�ֽ�
            toSend = 4096;
        }
        for (int i = 0; i < toSend; i++) {  // �����ݿ�����������
            buffer[i] = rawCharArray[sentLength + i];
        }
        int bytesSent = send(clientSocket, buffer, toSend, 0);  // ��������
        if (bytesSent == SOCKET_ERROR) {  // �������ʧ�ܣ����������Ϣ
            std::cerr << "Send failed" << std::endl;
            break;
        }
        sentLength += bytesSent;  // �����ѷ����ֽ���
    }

    // ������ͽ��
    printf("�������, �����ֽ���: %d", sentLength);

    // �رտͻ���Socket������WinSock
    closesocket(clientSocket);  // �ر�Socket����
    WSACleanup();  // ����WinSock��

    return 0;  // �������
}
