#include <iostream>
using namespace std;

#define Times 214748364
int better_toy(int &a, int &b, int step) {
  for (int i = 0; i < Times; i += step) {
    a++; //Can LLVM know that a = 198?
    if (a <= Times - 200) {
      b++; //Times - 2 
    } else {
      a = 0; //1
    }
  }
  return 0;
}

int main() {
  int step = 1;
  int a = Times - 198;
  int b = 0;
  int x = better_toy(a, b, step);
  cout << "a: "<< a << ", b: " <<b;
  return 0;
}