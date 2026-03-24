#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <memory>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::string str = "Hello, RISC-V C++ World!";
    std::pair<int, std::string> pair = {42, "test"};
    std::unique_ptr<int> ptr = std::make_unique<int>(123);
    
    return vec.size() + str.length() + *ptr;
}
