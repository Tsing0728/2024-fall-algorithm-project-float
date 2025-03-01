
//在第60行设置服务器的地址信息

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
alignas(32) char recieveFloatData[DATANUM];
alignas(32) char recieveCharData[DATANUM * 4];
alignas(32) float mergeFloatData[DATANUM * 2];
alignas(32) float max_values[THREADS_NUM];

SOCKET clientSocket;

// 初始化Socket，启动WinSock并建立与服务器的连接
void initSocket() {
    WSADATA wsaData;// 定义WSADATA结构体，用于存储WinSock的初始化信息    
    WORD version = MAKEWORD(2, 2);// 定义版本号

    // 调用WSAStartup初始化WinSock库，返回值wsResult表示初始化是否成功
    int wsResult = WSAStartup(version, &wsaData);

    // 检查WSAStartup是否成功
    if (wsResult != 0) {
        std::cerr << "Can't start WinSock, Err #" << wsResult << std::endl;// 如果初始化失败，输出错误信息并返回
        return;
    }
    // 创建客户端Socket，使用IPv4地址(AF_INET)、TCP协议(SOCK_STREAM)
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    // 设置服务器的地址信息
    sockaddr_in hint;
    hint.sin_family = AF_INET;   // 地址族为IPv4
    hint.sin_port = htons(8899); // 端口号为8899，将端口号转换为网络字节顺序
    inet_pton(AF_INET, "127.0.0.1", &hint.sin_addr); // 127.0.0.1

    // 使用连接函数与指定的服务器进行连接
    connect(clientSocket, (sockaddr*)&hint, sizeof(hint));
    // 输出连接成功信息
    std::cout << "Connected to server" << std::endl;
}

// 向服务器发送消息并接收服务器的响应
void sendMessage(std::string message) {
    if (send(clientSocket, message.c_str(), message.length(), 0) == SOCKET_ERROR) {
        // 如果发送失败，输出错误信息
        std::cerr << "Can't send data, Err #" << WSAGetLastError() << std::endl;
    }
    // 定义缓冲区，用于接收服务器返回的数据
    char buffer[4096];
    int bytesReceived;// 定义一个变量来存储接收到的字节数
    memset(buffer, 0, sizeof buffer);// 清空缓冲区，确保没有残留数据
    bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);// 从服务器接收数据，返回接收到的字节数

    // 检查接收结果
    if (bytesReceived == 0) {
        // 如果返回0，表示服务器关闭了连接
        std::cerr << "Connection closed by server" << std::endl;
    }
    else if (bytesReceived == SOCKET_ERROR) {
        // 如果返回SOCKET_ERROR，表示接收失败，输出错误信息
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;
    }

    // 输出接收到的数据
    std::cout << "Received: " << buffer << std::endl;
}

// 关闭Socket连接并清理WinSock资源
void close() {
    closesocket(clientSocket);
    WSACleanup();
}


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



// 用于打印部分数组（调试用）
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
    //  初始化Socket
    initSocket(); // 初始化网络通信Socket，建立与服务器的连接。

    // 数据初始化
    for (size_t i = 0; i < DATANUM; i++) { // 初始化rawFloatData数组
        rawFloatData[i] = log(sqrt(float(i + 1))); // 数据初始化为log(sqrt(i+1))的形式
    }
    //sendMessage(std::to_string(DATANUM)); // 发送数据量信息给服务器

    // 初始化计时器
    _LARGE_INTEGER time_start, time_over_1, time_over_2, time_over_3, time_over_4; // 定义时间点变量
    LARGE_INTEGER f; // 定义计时器频率
    QueryPerformanceFrequency(&f); // 获取计时器频率
    QueryPerformanceCounter(&time_start); // 开始计时
    sendMessage(std::to_string(DATANUM)); // 发送数据量信息给服务器

    //求和
    float SumResult = 0.0f;
    SumResult = sumOpenMp_AVX(rawFloatData, DATANUM); // 使用OpenMP和AVX进行加速求和
    QueryPerformanceCounter(&time_over_1); // 记录求和结束时间

    //最大值
    float MaxResult = 0.0f;
    MaxResult = maxOpenMp_AVX(rawFloatData, DATANUM); // 使用OpenMP和AVX进行加速求最大值
    QueryPerformanceCounter(&time_over_2); // 记录最大值计算结束时间

    // 堆排序
    heapSort_AVX(rawFloatData, DATANUM); // 使用AVX进行加速堆排序
    QueryPerformanceCounter(&time_over_3); // 记录排序结束时间

    cout << SumResult << endl; // 输出求和结果
    cout << MaxResult << endl; // 输出最大值结果

    // 将rawFloatData复制到mergeFloatData中,便于后续合并
    for (size_t i = 0; i < DATANUM; i++) {
        mergeFloatData[i] = rawFloatData[i];
    }

    // 接收网络数据
    char buffer[4096]; // 定义缓冲区
    int bytesReceived; // 接收的字节数
    memset(buffer, 0, sizeof buffer); // 清空缓冲区，确保缓冲区是空的，防止脏数据影响后续解析
    bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0); // 接收数据：从指定的 clientSocket 中读取数据到 buffer

    if (bytesReceived == 0) {
        std::cerr << "Connection closed by server" << std::endl; // 服务器关闭连接
    }
    else if (bytesReceived == SOCKET_ERROR) {
        std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl; // 接收数据错误
    }


    std::cout << "Received: " << buffer << std::endl; // 打印接收到的数据

    if (send(clientSocket, "Received Data!", 14, 0) == SOCKET_ERROR) {
        std::cerr << "Can't send data, Err #" << WSAGetLastError() << std::endl; // 发送确认消息失败
    }

    //  解析接收到的数据
    string info(buffer); // 将缓冲区数据转换为字符串，便于后续处理。
    vector<float> result = Stringsplit(info, '|'); // 使用'|'分割字符串为浮点数向量，过来的分别是总和、最大值、排序判断结果

    SumResult += result[0]; // 算最终总和
    MaxResult = max(MaxResult, result[1]); // 比较两个最大值，更新最终最大值

    //接下来，接收对方排好序的数组
    int receivedLength = 0;
    do {
        memset(buffer, 0, sizeof buffer); // 清空缓冲区
        bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0); // 接收数据
        for (size_t i = 0; i < bytesReceived; i++) {
            recieveCharData[i + receivedLength] = buffer[i]; // 将接收到的数据存储到recieveCharData，则每次 recv 的数据追加到一个连续的内存区域，确保所有数据都被正确存储
        }
        receivedLength += bytesReceived; // 更新已接收字节数
    } while (bytesReceived == 4096); // 如果缓冲区满，说明可能还有数据没收到，继续接收


    printf("收到字节长度：%d ，", receivedLength);

    // 数据合并与重排序
    std::cout << "准备重排序..." << std::endl;

    memcpy(recieveFloatData, recieveCharData, DATANUM); // 将接收到的字符数据转换为浮点数据
    _memccpy(mergeFloatData, rawFloatData, 0, DATANUM); // 将原始数据复制到合并数组中
    _memccpy(mergeFloatData + DATANUM, recieveFloatData, 0, DATANUM); // 将接收到的数据追加到合并数组中
   

    heapSort_AVX(mergeFloatData, DATANUM * 2); //堆排序
    std::cout << "重排序结束" << std::endl;
    QueryPerformanceCounter(&time_over_4);	//计时结束


    //时间统计
    std::cout << "Time Consumed:" << float(time_over_3.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    cout << SumResult << endl;
    cout << MaxResult << endl;
    std::cout << "Sum Time Consumed:" << float(time_over_1.QuadPart - time_start.QuadPart) / f.QuadPart << endl;
    std::cout << "Max Time Consumed:" << float(time_over_2.QuadPart - time_over_1.QuadPart) / f.QuadPart << endl;
    std::cout << "Sort Time Consumed:" << float(time_over_3.QuadPart - time_over_2.QuadPart) / f.QuadPart << endl;
    std::cout << "Master&Node Finish Time Consumed:" << float(time_over_4.QuadPart - time_start.QuadPart) / f.QuadPart << endl;


    bool checkflag = checkSort(mergeFloatData, DATANUM * 2); //重排序检查
    printf("检查排序：%d", checkflag);
    cout << endl;
    //printArray(rawFloatData, 20);  // 只打印前20个元素
    return 0;
}