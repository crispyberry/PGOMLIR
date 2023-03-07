// #include <iostream>
// using namespace std;
// This is more simple better toy, since we ignore the
// Interprocedural-Analysis(we don't care functioncall)
#define Times 6
int better_toy() {
  int step =1;
  int a = 2;
  int b = 0;
  for (int i = 0; i < Times; i += step) {
    a++; // a=1
    if (a+b <= 4) {
      b++; // b=4
    } else {
      a = 0; 
    }
  }
  return a + b;
}

int main() {
  int x = better_toy();
  //cout << "a:"<<a<<"b:" << b;
  return 0;
}