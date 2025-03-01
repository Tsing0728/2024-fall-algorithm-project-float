#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <iostream>

#pragma comment(lib, "ws2_32.lib")

SOCKET clientSocket;

void initSocket() {
    WSADATA wsaData;
    WORD version = MAKEWORD(2, 2);
    int wsResult = WSAStartup(version, &wsaData);
    if (wsResult != 0) {
        std::cerr << "Can't start WinSock, Err #" << wsResult << std::endl;
        return;
    }

    clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in hint;
    hint.sin_family = AF_INET;
    hint.sin_port = htons(8899);
    inet_pton(AF_INET, "127.0.0.1", &hint.sin_addr);

    connect(clientSocket, (sockaddr*)&hint, sizeof(hint));

    std::cout << "Connected to server" << std::endl;
}

int main() {
    initSocket(); //初始化Socket

    char buffer[4096];
    int bytesReceived;
    while (true) {
        std::cout << "Enter message to send: ";
        std::cin.getline(buffer, sizeof(buffer));

        if (strcmp(buffer, "exit\n") == 0) {
            std::cout << "Closing connection..." << std::endl;
            break;
        }

        if (send(clientSocket, buffer, strlen(buffer), 0) == SOCKET_ERROR) {
            std::cerr << "Can't send data, Err #" << WSAGetLastError() << std::endl;
            break;
        }

        memset(buffer, 0, sizeof buffer);
        bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
        if (bytesReceived == 0) {
            std::cerr << "Connection closed by server" << std::endl;
            break;
        } else if (bytesReceived == SOCKET_ERROR) {
            std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;
            break;
        }

        std::cout << "Received: " << buffer << std::endl;
    }

    closesocket(clientSocket);
    WSACleanup();
    return 0;
}