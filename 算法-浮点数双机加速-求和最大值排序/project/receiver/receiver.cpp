#include <WinSock2.h>
#include <iostream>

#pragma comment(lib, "ws2_32.lib")

[[noreturn]] int main() {
    WSADATA wsaData;
    WORD version = MAKEWORD(2, 2);
    int wsResult = WSAStartup(version, &wsaData);
    if (wsResult != 0) {
        std::cerr << "Can't start WinSock, Err #" << wsResult << std::endl;
        return -1;
    }

    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in hint;
    hint.sin_family = AF_INET;
    hint.sin_port = htons(8899);
    hint.sin_addr.S_un.S_addr = INADDR_ANY;

    bind(serverSocket, (sockaddr*)&hint, sizeof(hint));
    listen(serverSocket, 5);

    sockaddr_in client;
    int clientSize = sizeof(client);
    SOCKET clientSocket = accept(serverSocket, (sockaddr*)&client, &clientSize);



    char buffer[4096];
    int bytesReceived;
    while (true) {
        memset(buffer, 0, 4096);
        bytesReceived = recv(clientSocket, buffer, 4096, 0);
        if (bytesReceived == 0) {
            std::cerr << "Connection closed by Client" << std::endl;
            break;
        } else if (bytesReceived == SOCKET_ERROR) {
            std::cerr << "Can't receive data, Err #" << WSAGetLastError() << std::endl;
            break;
        } else if (bytesReceived < 4096) {
            std::cout << "Received: " << buffer << std::endl;
            send(clientSocket, "Received Data!", 14, 0);
        } else {
            std::cout << "Received: " << buffer << std::endl;
        }
    }
    closesocket(clientSocket);
    WSACleanup();
    return 0;
}