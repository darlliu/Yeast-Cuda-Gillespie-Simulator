#include "cuda.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
using namespace std;

#define BDIM 128
#define BLOCKS 16
#define LOOPS 3200
#define LOG 0

#include "ran.cu"
#include "cudahelper.hpp"
#include "simplepng.hpp"
