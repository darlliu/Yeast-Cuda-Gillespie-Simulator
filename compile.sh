module load cuda
nvcc -O3 -o main ./main.cu -L/dahome/yl40/libpng/lib -lpng
