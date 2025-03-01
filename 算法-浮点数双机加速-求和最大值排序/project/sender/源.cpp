
//�ڵ�60�����÷������ĵ�ַ��Ϣ

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
alignas(32) char recieveFloatData[DATANUM];
alignas(32) char recieveCharData[DATANUM * 4];
alignas(32) float mergeFloatData[DATANUM * 2];
alignas(32) float max_values[THREADS_NUM];

SOCKET clientSocket;

// ��ʼ��Socket������WinSock�������������������
void initSocket() {
    WSADATA wsaData;// ����WSADATA�ṹ�壬���ڴ洢WinSock�ĳ�ʼ����Ϣ    
    WORD version = MAKEWORD(2, 2);// ����汾��

    // ����WSAStartup��ʼ��WinSock�⣬����ֵwsResult��ʾ��ʼ���Ƿ�ɹ�
    int wsResult = WSAStartup(version, &wsaData);

    // ���WSAStartup�Ƿ�ɹ�
    if (wsResult != 0) {
        std::cerr << "Can't start WinSock, Err #" << wsResult << std::endl;// �����ʼ��ʧ�ܣ����������Ϣ������
        return;
    }
    // �����ͻ���Socket��ʹ��IPv4��ַ(AF_INET)��TCPЭ��(SOCK_STREAM)
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    // ���÷������ĵ�ַ��Ϣ
    sockaddr_in hint;
    hint.sin_family = AF_INET;   // ��ַ��ΪIPv4
    hint.sin_port = htons(8899); // �˿ں�Ϊ8899�����˿ں�ת��Ϊ�����ֽ�˳��
    inet_pton(AF_INET, "127.0.0.1", &hint.sin_addr); // 127.0.0.1

    // ʹ�����Ӻ�����ָ���ķ�������������
    connect(clientSocket, (sockaddr*)&hint, sizeof(hint));
    // ������ӳɹ���Ϣ
    std::cout << "Connected to server" << std::endl;
}

// �������������Ϣ�����շ���������Ӧ
void sendMessage(std::string message) {
    if (send(clientSocket, message.c_str(), message.length(), 0) == SOCKET_ERROR) {
        // �������ʧ�ܣ����������Ϣ
        std::cerr << "Can't send data, Err #" << WSAGetLastError() << std::endl;
    }
    // ���建���������ڽ��շ��������ص�����
    char buffer[4096];
    int bytesReceived;// ����һ���������洢���յ����ֽ���
    memset(buffer, 0, sizeof buffer);// ��ջ�������ȷ��û�в�������
    bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);// �ӷ������������ݣ����ؽ��յ����ֽ���

    // �����ս��
    if (bytesReceived == 0) {
        // �������0����ʾ�������ر�������
        std::cerr << "Connection closed by server" << std::endl;
    }
    else if (bytesReceived == SOCKET_ERROR) {
        // �������SOCKET_ERROR����ʾ����ʧ�ܣ����������Ϣ
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;
    }

    // ������յ�������
    std::cout << "Received: " << buffer << std::endl;
}

// �ر�Socket���Ӳ�����WinSock��Դ
void close() {
    closesocket(clientSocket);
    WSACleanup();
}


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
#pragma omp critical
        {
            mySwap(arr[0], arr[i]);
            heapify_AVX(arr, 0, i);
        }
    }
}




bool checkSort(const float data[], const int len) {
    for (int i = 0; i < len - 1; i++) {
        if (data[i + 1] < data[i]) {
            printf("Error at index %d: %f > %f\n", i, data[i], data[i + 1]);
            return false;
        }
    }
    return true;
}



// ���ڴ�ӡ�������飨�����ã�
void printArray(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}
//

vector<float> Stringsplit(string str, char split) {
    vector<float> result;
    istringstream iss(str);
    string token;
    while (getline(iss, token, split)) {
        result.push_back(stof(token));
    }
    return result;
}


int main() {
    //  ��ʼ��Socket
    initSocket(); // ��ʼ������ͨ��Socket������������������ӡ�

    // ���ݳ�ʼ��
    for (size_t i = 0; i < DATANUM; i++) { // ��ʼ��rawFloatData����
        rawFloatData[i] = log(sqrt(float(i + 1))); // ���ݳ�ʼ��Ϊlog(sqrt(i+1))����ʽ
    }
    //sendMessage(std::to_string(DATANUM)); // ������������Ϣ��������

    // ��ʼ����ʱ��
    _LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3, time_over_4; // ����ʱ������
    LARGE_INTEGER f; // �����ʱ��Ƶ��
    QueryPerformanceFrequency(&f); // ��ȡ��ʱ��Ƶ��
    QueryPerformanceCounter(&time_start); // ��ʼ��ʱ
    sendMessage(std::to_string(DATANUM)); // ������������Ϣ��������

    //���
    float SumResult = 0.0f;
    SumResult = sumOpenMp_AVX(rawFloatData, DATANUM); // ʹ��OpenMP��AVX���м������
    QueryPerformanceCounter(&time_over_1); // ��¼��ͽ���ʱ��

    //���ֵ
    float MaxResult = 0.0f;
    MaxResult = maxOpenMp_AVX(rawFloatData, DATANUM); // ʹ��OpenMP��AVX���м��������ֵ
    QueryPerformanceCounter(&time_over_2); // ��¼���ֵ�������ʱ��

    // ������
    heapSort_AVX(rawFloatData, DATANUM); // ʹ��AVX���м��ٶ�����
    QueryPerformanceCounter(&time_over_3); // ��¼�������ʱ��

    cout << SumResult << endl; // �����ͽ��
    cout << MaxResult << endl; // ������ֵ���

    // ��rawFloatData���Ƶ�mergeFloatData��,���ں����ϲ�
    for (size_t i = 0; i < DATANUM; i++) {
        mergeFloatData[i] = rawFloatData[i];
    }

    // ������������
    char buffer[4096]; // ���建����
    int bytesReceived; // ���յ��ֽ���
    memset(buffer, 0, sizeof buffer); // ��ջ�������ȷ���������ǿյģ���ֹ������Ӱ���������
    bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0); // �������ݣ���ָ���� clientSocket �ж�ȡ���ݵ� buffer

    if (bytesReceived == 0) {
        std::cerr << "Connection closed by server" << std::endl; // �������ر�����
    }
    else if (bytesReceived == SOCKET_ERROR) {
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl; // �������ݴ���
    }


    std::cout << "Received: " << buffer << std::endl; // ��ӡ���յ�������

    if (send(clientSocket, "Received Data!", 14, 0) == SOCKET_ERROR) {
        std::cerr << "Can't send data, Err #" << WSAGetLastError() << std::endl; // ����ȷ����Ϣʧ��
    }

    //  �������յ�������
    string info(buffer); // ������������ת��Ϊ�ַ��������ں�������
    vector<float> result = Stringsplit(info, '|'); // ʹ��'|'�ָ��ַ���Ϊ�����������������ķֱ����ܺ͡����ֵ�������жϽ��

    SumResult += result[0]; // �������ܺ�
    MaxResult = max(MaxResult, result[1]); // �Ƚ��������ֵ�������������ֵ

    //�����������նԷ��ź��������
    int receivedLength = 0;
    do {
        memset(buffer, 0, sizeof buffer); // ��ջ�����
        bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0); // ��������
        for (size_t i = 0; i < bytesReceived; i++) {
            recieveCharData[i + receivedLength] = buffer[i]; // �����յ������ݴ洢��recieveCharData����ÿ�� recv ������׷�ӵ�һ���������ڴ�����ȷ���������ݶ�����ȷ�洢
        }
        receivedLength += bytesReceived; // �����ѽ����ֽ���
    } while (bytesReceived == 4096); // �������������˵�����ܻ�������û�յ�����������


    printf("�յ��ֽڳ��ȣ�%d ��", receivedLength);

    // ���ݺϲ���������
    std::cout << "׼��������..." << std::endl;

    memcpy(recieveFloatData, recieveCharData, DATANUM); // �����յ����ַ�����ת��Ϊ��������
    _memccpy(mergeFloatData, rawFloatData, 0, DATANUM); // ��ԭʼ���ݸ��Ƶ��ϲ�������
    _memccpy(mergeFloatData + DATANUM, recieveFloatData, 0, DATANUM); // �����յ�������׷�ӵ��ϲ�������
   

    heapSort_AVX(mergeFloatData, DATANUM * 2); //������
    std::cout << "���������" << std::endl;
    QueryPerformanceCounter(&time_over_4);	//��ʱ����


    //ʱ��ͳ��
    std::cout << "Time Consumed:" << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    cout << SumResult << endl;
    cout << MaxResult << endl;
    std::cout << "Sum Time Consumed:" << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    std::cout << "Max Time Consumed:" << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << endl;
    std::cout << "Sort Time Consumed:" << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << endl;
    std::cout << "Master&Node Finish Time Consumed:" << float(time_over_4.QuadPart - time_start.QuadPart) / f.QuadPart << endl;


    bool checkflag = checkSort(mergeFloatData, DATANUM * 2); //��������
    printf("�������%d", checkflag);
    cout << endl;
    //printArray(rawFloatData, 20);  // ֻ��ӡǰ20��Ԫ��
    return 0;
}