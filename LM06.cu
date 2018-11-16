#include <iostream>

using namespace std;

// CUDA Kernel
//Performs matrix multiplication A * B = Out
//Note that aWidth must equal bHeight for the multiplication to succeed
//Thus we have summarily done away with the latter to remove temptation
__global__ void matrixMultiply(double *matrixA, double *matrixB, double* matrixOut, int aHeight, 
                                int aWidth, int bWidth) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = gridDim.x * gridDim.y * blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
    double sum = 0;
    // check to see if we are inside our problem space
    if (tid < aHeight * bWidth) {
        // calculate row and col that we are going to compute
        // loop over A & B at the same time since A is row major and B is column major
        for (int ndx = 0; ndx < aWidth; ndx++) {
            double lhs = *(matrixA + row*aWidth + ndx);
            double rhs = *(matrixB + col*aWidth + ndx);
            //Accumulate result
            sum += lhs * rhs; 
        }
        // store in matrix
        *(matrixOut + tid) = sum;
    }
    
}

void fillMatrix(double *target, int targetSize) {
    for (double ndx = 0; ndx < targetSize; ndx += 1) {
        *target = ndx;
        target++;
    }
}

void printMatrixRowMaj(double *target, int numRows, int numCols) {
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            std::cout << *(target + row * numCols + col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std:: endl;
}

void printMatrixColMaj(double *target, int numRows, int numCols) {
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            std::cout << *(target + col * numRows + row) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std:: endl;
}

int main() {
    int aHeight = 3;    //num of rows in A
    int aWidth = 2;     //num of cols in A
    int bHeight = 2;    //num of rows in B - this must be the same as aWidth for AB to work
    int bWidth = 3;     //num of cols in B
    double *dev_matrixA, *dev_matrixB, *dev_matrixOut;
    cudaEvent_t start, stop;
    float milliseconds; //how long did we take to do things?

    bHeight = aWidth;   //Let's just make sure

    //allocate space
    double* matrixA = (double * )malloc(sizeof (double) * aHeight * aWidth);
    double* matrixB = (double * )malloc(sizeof (double) * bHeight * bWidth);        //The operand matrices
    double* matrixOut = (double * )malloc(sizeof (double) * aHeight * bWidth);      //The result matrix

    //fill operands
    fillMatrix(matrixA, aHeight * aWidth);
    fillMatrix(matrixB, bHeight * bWidth);

    //setup memory shit
    cudaMalloc((void**)&dev_matrixA, (aHeight * aWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixB, (bHeight * bWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixOut, (aHeight * bWidth) * sizeof(double));

    // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(dev_matrixA, matrixA, aHeight * aWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixB, matrixB, bHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixOut, matrixOut, aHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice);

    //start timer event
    cudaEventRecord(start);
    //call kernel
    dim3 threadsPerBlock (1, 32);
    dim3 blocks (32, 32);
    matrixMultiply<<<1,threadsPerBlock>>>(dev_matrixA, dev_matrixB, dev_matrixOut, aHeight, aWidth, bWidth);
    //stop timer event
    cudaEventRecord(stop);

    //get result
    cudaMemcpy(matrixOut, dev_matrixOut, aHeight * bWidth * sizeof(double), cudaMemcpyDeviceToHost);
    
    //calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    //free memory
    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixOut);

    printMatrixRowMaj(matrixA, aHeight, aWidth);
    printMatrixColMaj(matrixB, bHeight, bWidth);
    printMatrixRowMaj(matrixOut, aHeight, bWidth);



    return 0;
}