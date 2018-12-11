#include <iomanip>
#include <iostream>

using namespace std;

// CUDA Kernel
//Performs matrix multiplication A * B = Out
//Note that aWidth must equal bHeight for the multiplication to succeed
//Thus we have summarily done away with the latter to remove temptation
//This kernel assumes that A is row major and B is column major
__global__ void matrixMultiply(double *matrixA, double *matrixB, double* matrixOut, 
                                int aHeight, int aWidth, int bWidth) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = row * bWidth + col;
 
    double sum = 0;
    // check to see if we are inside our problem space
    if (row < aHeight && col < bWidth) {
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

// CUDA Kernel
//Performs matrix multiplication A + B = Out
//Both operand matrices must be square and have the same dimension
__global__ void matrixAdd(double *matrixA, double *matrixB, double* matrixOut, 
                                int aHeight, int aWidth) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = row * aWidth + col;
 
    // check to see if we are inside our problem space
    if (row < aHeight && col < aWidth) {
        *(matrixOut + tid) = *(matrixA + tid) + *(matrixB + tid);    
    }   
}

void fillMatrix(double *target, int targetSize) {
    for (double ndx = 0; ndx < targetSize; ndx += 1) {
        *target = (int)ndx % 100;
        target++;
    }
}

void printMatrixRowMaj(double *target, int numRows, int numCols) {
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            std::cout << std::setw(7) << *(target + row * numCols + col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std:: endl;
}

void printMatrixColMaj(double *target, int numRows, int numCols) {
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            std::cout << std::setw(7) << *(target + col * numRows + row) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std:: endl;
}

int main() {
    int aHeight = 32;    //num of rows in A
    const int aWidth = 32;     //num of cols in A
    const int bHeight = 32;    //num of rows in B - this must be the same as aWidth for AB to work
    int bWidth = 32;     //num of cols in B
    //For simplicity's sake we will assume that C has the same dimensions as C and D has the same dimensions as B
    double *dev_matrixA, *dev_matrixB, *dev_matrixProd1;
    double *dev_matrixC, *dev_matrixD, *dev_matrixProd2;
    double *dev_matrixSum;
    cudaStream_t stream1, stream2;
    cudaEvent_t start, stop, mult1done, mult2done;
    float milliseconds; //how long did we take to do things?
    
    //bHeight = aWidth;   //Let's just make sure

    //allocate space
    double* matrixA = (double * )malloc(sizeof (double) * aHeight * aWidth);
    double* matrixB = (double * )malloc(sizeof (double) * bHeight * bWidth);        //The operand matrices for the first mult
    double* matrixC = (double * )malloc(sizeof (double) * aHeight * aWidth);
    double* matrixD = (double * )malloc(sizeof (double) * bHeight * bWidth);        //The operand matrices for the second mult

    double* matrixProd1 = (double * )malloc(sizeof (double) * aHeight * bWidth);      //The result matrix of the first mult
    double* matrixProd2 = (double * )malloc(sizeof (double) * aHeight * bWidth);      //The result matrix of the second mult
    double* matrixSum = (double * )malloc(sizeof (double) * aHeight * bWidth);      //The result matrix

    //fill operands
    fillMatrix(matrixA, aHeight * aWidth);
    fillMatrix(matrixB, bHeight * bWidth);
    fillMatrix(matrixC, aHeight * aWidth);
    fillMatrix(matrixD, bHeight * bWidth);

    //setup memory on device
    cudaMalloc((void**)&dev_matrixA, (aHeight * aWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixB, (bHeight * bWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixProd1, (aHeight * bWidth) * sizeof(double));

    cudaMalloc((void**)&dev_matrixC, (aHeight * aWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixD, (bHeight * bWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixProd2, (aHeight * bWidth) * sizeof(double));
    
    cudaMalloc((void**)&dev_matrixSum, (aHeight * bWidth) * sizeof(double));
    

    //Set up problem space dimensions
    //dim3 threadsPerBlock (bWidth, aHeight);
    dim3 threadsPerBlock (32, 32);
    dim3 blocks (1,1);

    //Create streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&mult1done);
    cudaEventCreate(&mult2done);
   
    cudaEventRecord(start); 

    //Load the operands for the first multiplication
    cudaMemcpyAsync(dev_matrixA, matrixA, aHeight * aWidth * sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_matrixB, matrixB, bHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_matrixProd1, matrixProd1, aHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice, stream1);

    //call multiply kernel in stream 1
    matrixMultiply<<<blocks,threadsPerBlock, 0, stream1>>>(dev_matrixA, dev_matrixB, dev_matrixProd1, aHeight, aWidth, bWidth);
    
    //While the first multiply is running, load the operands for the second multiplication
    cudaMemcpyAsync(dev_matrixC, matrixC, aHeight * aWidth * sizeof(double), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_matrixD, matrixD, bHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_matrixProd2, matrixProd2, aHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice, stream2);

    //call multiply kernel in stream 2
    matrixMultiply<<<blocks,threadsPerBlock, 0, stream2>>>(dev_matrixC, dev_matrixD, dev_matrixProd2, aHeight, aWidth, bWidth);

    //get result from device
    cudaMemcpyAsync(matrixProd1, dev_matrixProd1, aHeight * bWidth * sizeof(double), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(matrixProd2, dev_matrixProd2, aHeight * bWidth * sizeof(double), cudaMemcpyDeviceToHost, stream2);
    
    //Ensure that both stream1 and stream2 are done
    //cudaEventRecord(mult1done, stream1); //<-- If we use stream1 to perform the addition, we do not need this event. Stream1 won't continue until it's finished its memcpy
    cudaEventRecord(mult2done, stream2);

    //Make sure the second multiplication is done before continuing
    cudaStreamWaitEvent(stream1, mult2done, 0);

    //Copy the multiplication results to the device 
    cudaMemcpyAsync(dev_matrixProd1, matrixProd1, aHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_matrixProd2, matrixProd2, aHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_matrixSum, matrixSum, aHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice, stream1);
    
    //Call the addition kernel
    matrixAdd<<<blocks,threadsPerBlock, 0, stream1>>>(dev_matrixProd1, dev_matrixProd2, dev_matrixSum, aHeight, aWidth);

    //Get the result
    cudaMemcpyAsync(matrixSum, dev_matrixSum, aHeight * bWidth * sizeof(double), cudaMemcpyDeviceToHost, stream1);
    
    //calculate time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    //Test our calculation
    printMatrixRowMaj(matrixA, aHeight, aWidth);
    printMatrixColMaj(matrixB, bHeight, bWidth);
    printMatrixRowMaj(matrixProd1, aHeight, bWidth);
    printMatrixRowMaj(matrixProd2, aHeight, bWidth);
    printMatrixRowMaj(matrixSum, aHeight, bWidth);

    //free memory
    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixProd1);
    cudaFree(dev_matrixC);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixProd1);
    cudaFree(dev_matrixSum);

    std::cout << "It took " << milliseconds << " milliseconds to complete.\n";

    return 0;
}
