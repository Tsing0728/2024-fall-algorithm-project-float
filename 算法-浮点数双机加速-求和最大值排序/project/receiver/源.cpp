#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <omp.h>        // OpenMP 支持
#include <algorithm>
#include <thread>       // 多线程支持（可能未必需要）
#include <time.h>
#include <stdlib.h>
#include <windows.h>    // Windows API 支持 (例如 QueryPerformanceCounter)
#include <immintrin.h>  // AVX 指令集支持
#include <float.h>      // 用于 FLT_MAX
#include <cfloat>

using namespace std;

#pragma comment(lib, "ws2_32.lib")

#define MAX_THREADS 64
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*这个数值是总数据量*/

//sum & max
#define THREADS_NUM 64


#define ALIGNMENT 32   // AVX 使用 256 位寄存器，内存对齐 32 字节

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

// 求和函数 (OpenMP 加速)
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
        float local_result = temp[0] + temp[1] + temp[2] + temp[3] +
            temp[4] + temp[5] + temp[6] + temp[7];

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


float maxOpenMp(const float data[], const int len) {
    float max_values[THREADS_NUM] = { 0 };  // 存储每个线程的最大值
    int chunk_size = len / THREADS_NUM;  // 每个线程处理的数据量

#pragma omp parallel num_threads(THREADS_NUM)
    {
        int thread_id = omp_get_thread_num();  // 获取当前线程 ID
        int start = thread_id * chunk_size;    // 每个线程的起始索引
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size; // 最后一个线程处理剩余部分

        float local_max = data[start];
        for (int i = start + 1; i < end; i++) {
            if (data[i] > local_max) {
                local_max = data[i];
            }
        }
        max_values[thread_id] = local_max;  // 保存线程局部最大值
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



float maxOpenMp_AVX(const float data[], const int len) {
    float max_values[THREADS_NUM] = { -FLT_MAX };  // 存储每个线程的最大值
    int chunk_size = len / THREADS_NUM;  // 每个线程处理的数据量

#pragma omp parallel num_threads(THREADS_NUM)
    {
        int thread_id = omp_get_thread_num();  // 获取当前线程 ID
        int start = thread_id * chunk_size;    // 每个线程的起始索引
        int end = (thread_id == THREADS_NUM - 1) ? len : start + chunk_size; // 最后一个线程处理剩余部分

        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);  // 初始化 AVX 最大值向量

        // 使用 AVX 处理 8 个浮点数一组
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

// 自定义交换函数，用于交换两个变量的值
// 模板使其支持不同类型的数据交换
template<typename T>
void mySwap(T& a, T& b) {
    T temp = a; // 将 a 的值存储到临时变量 temp 中
    a = b;      // 将 b 的值赋给 a
    b = temp;   // 将 temp（原 a 的值）赋给 b
}

// 堆化函数
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
            idx = largest;  // 继续下沉
        }
        else {
            break;  // 如果没有交换，结束
        }
    }
}


void heapSort_AVX(float* arr, int len) {
    // 构建最大堆
    for (int i = len / 2 - 1; i >= 0; i--) {
        heapify_AVX(arr, i, len);
    }

    // 提取元素
    for (int i = len - 1; i > 0; i--) {
        mySwap(arr[0], arr[i]);
        heapify_AVX(arr, 0, i);
    }
}

//排序测试
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
    // 初始化WinSock
    WSADATA wsaData;
    WORD version = MAKEWORD(2, 2); 
    int wsResult = WSAStartup(version, &wsaData);  // 初始化WinSock库
    if (wsResult != 0) {  // 检查初始化是否成功
        std::cerr << "Can't start WinSock, Err #" << wsResult << std::endl;  // 输出错误信息
        return -1;  // 如果初始化失败，退出程序
    }

    // 创建服务器Socket，使用IPv4和TCP协议
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);  // 创建TCP服务器Socket

    
    sockaddr_in hint;
    hint.sin_family = AF_INET;  
    hint.sin_port = htons(8899);  // 设置端口号8899（将端口号转换为网络字节顺序）
    hint.sin_addr.S_un.S_addr = INADDR_ANY;  // 设置IP地址为INADDR_ANY，表示接受任何客户端的连接

    // 绑定服务器Socket到指定的IP和端口
    bind(serverSocket, (sockaddr*)&hint, sizeof(hint));  // 将服务器Socket绑定到指定地址和端口

    listen(serverSocket, 5);  // 设置监听，最多同时处理5个连接请求

    // 接受客户端连接
    sockaddr_in client;
    int clientSize = sizeof(client);  // 客户端地址的大小
    SOCKET clientSocket = accept(serverSocket, (sockaddr*)&client, &clientSize);  // 接受客户端连接

    // 接收客户端发来的数据
    char buffer[4096];  // 定义缓冲区用于接收数据
    int bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);  // 从客户端接收数据
    if (bytesReceived == 0) {  // 如果接收到的数据长度为0，表示客户端关闭了连接
        std::cerr << "Connection closed by Client" << std::endl;  // 输出错误信息
        return -1;  // 退出程序
    }
    else if (bytesReceived == SOCKET_ERROR) {  // 如果接收数据出错
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;  // 输出错误信息
        return -1;
    }

    // 输出接收到的数据
    std::cout << "Received: " << buffer << std::endl;

    // 向客户端发送响应消息
    send(clientSocket, "Received Data!", 14, 0);  // 发送"Received Data!"给客户端

    // 处理接收到的数据，计算初始值
    int start = atoi(buffer);  // 将接收到的字符串转为整数

    for (size_t i = 0; i < DATANUM; i++) {  // 数据初始化：从后64000000开始
        rawFloatData[i] = log(sqrt(float(start + i + 1)));  // 计算并填充rawFloatData数组
    }

    // 初始化性能计时器
    _LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3;  // 定义计时变量
    LARGE_INTEGER f;  // 计时器频率
    QueryPerformanceFrequency(&f);  // 获取计时器的频率
    QueryPerformanceCounter(&time_start);  // 开始计时

    // 调用speedup求和函数
    float SumResult = 0.0f;
    SumResult = sumOpenMp_AVX(rawFloatData, DATANUM);  
    QueryPerformanceCounter(&time_over_1);  // 记录求和结束时间

    // 调用speedup求最大值函数
    float MaxResult = 0.0f;
    MaxResult = maxOpenMp_AVX(rawFloatData, DATANUM); 
    QueryPerformanceCounter(&time_over_2);  // 记录最大值计算结束时间

    // 对数据进行堆排序
    heapSort_AVX(rawFloatData, DATANUM); 
    QueryPerformanceCounter(&time_over_3);  // 记录排序结束时间

    // 输出性能测试结果
    std::cout << "Time Consumed:" << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << std::endl;
    std::cout << SumResult << std::endl;  // 输出求和结果
    std::cout << MaxResult << std::endl;  // 输出最大值结果
    std::cout << "Sum Time Consumed:" << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << std::endl;
    std::cout << "Max Time Consumed:" << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << std::endl;
    std::cout << "Sort Time Consumed:" << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << std::endl;

    // 检查排序是否正确
    bool checkflag;
    checkflag = checkSort(rawFloatData, 1000000);  // 检查排序结果是否正确
    printf("检查排序：%d", checkflag);  // 输出排序检查结果
    std::cout << std::endl;

    // 将三个结果封装为字符串并发送给客户端
    string dataString = std::to_string(SumResult) + "|" + std::to_string(MaxResult) + "|" + std::to_string(checkflag);
    send(clientSocket, dataString.c_str(), dataString.size(), 0);  // 发送处理结果给客户端



    // 接收客户端的反馈数据
    memset(buffer, 0, sizeof buffer);  // 清空缓冲区

    bytesReceived = recv(clientSocket, buffer, 4096, 0);  // 接收数据
    if (bytesReceived == 0) {  // 如果接收到的数据长度为0，表示客户端关闭了连接
        std::cerr << "Connection closed by server" << std::endl;
    }
    else if (bytesReceived == SOCKET_ERROR) {  // 如果接收数据出错
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;
    }
    std::cout << "Received: " << buffer << std::endl;



    // 将浮点数据数组转换为字符数组进行发送
    memcpy(rawCharArray, rawFloatData, DATANUM * 4);  // 将浮点数数组转换为字符数组

    // 发送rawFloatData数据
    int sentLength = 0;  // 已发送的字节数
    int charLength = DATANUM * 4;  //float 类型占 4 字节，算数据总字节数
    while (sentLength < charLength) {  // 循环发送数据
        int toSend = charLength - sentLength;  // 计算还需发送的字节数
        if (toSend > 4096) {  // 每次发送最大4096字节
            toSend = 4096;
        }
        for (int i = 0; i < toSend; i++) {  // 将数据拷贝到缓冲区
            buffer[i] = rawCharArray[sentLength + i];
        }
        int bytesSent = send(clientSocket, buffer, toSend, 0);  // 发送数据
        if (bytesSent == SOCKET_ERROR) {  // 如果发送失败，输出错误信息
            std::cerr << "Send failed" << std::endl;
            break;
        }
        sentLength += bytesSent;  // 更新已发送字节数
    }

    // 输出发送结果
    printf("发送完成, 发送字节数: %d", sentLength);

    // 关闭客户端Socket并清理WinSock
    closesocket(clientSocket);  // 关闭Socket连接
    WSACleanup();  // 清理WinSock库

    return 0;  // 程序结束
}
