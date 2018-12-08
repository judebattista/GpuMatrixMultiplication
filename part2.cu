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

__global__ void sharedMatrixMultiply(double *matrixA, double *matrixB, double* matrixOut, 
        int aHeight, int aWidth, int bWidth,
        double* sharedTestA, double* sharedTestB) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = row * bWidth + col;
    const int sharedWidth = 32;
    const int sharedHeight = 32;

    __shared__ double sharedA[sharedWidth * sharedHeight];
    __shared__ double sharedB[sharedWidth * sharedHeight];
   
    //figure out which rows of A and columns of B need to be loaded into the shared memory
    //This should be based off the TID for the output matrix
    //If we're in the first row of the output matrix, we need the first row of A
    //If we're in the first column of the output matrix, we need the first column of B
    //This correspondence seems to hold over the output space
    //The size of our block determines how many rows and columns we need to hold
      

     

    //TODO: 
    //Replace index math to work with multiple blocks.
    //aWidth needs to go. Needs to work in chunks of 32
    //Calculate which shared block we are in based on sharedWidth
    *(sharedA + row * sharedWidth + col) = *(matrixA + row * aWidth + col);
    *(sharedB + row * sharedWidth + col) = *(matrixB + row * aWidth + col); //Note: aWidth = bHeight
    __syncthreads();

    //Calculate which shared block our TID is in
    sharedBlockIdx = blockIdx.x;
    sharedBlockIdy = blockIdx.y;

    for(int ndx = 0; ndx < sharedHeight * sharedWidth; ndx++) {
        *(sharedTestA + ndx) = *(sharedA + ndx);
        *(sharedTestB + ndx) = *(sharedB + ndx);
    }

    double sum = 0;
    double lhs = 0;
    double rhs = 0;
    // check to see if we are inside our problem space
    if (row < aHeight && col < bWidth) {
        // calculate row and col that we are going to compute
        // loop over A & B at the same time since A is row major and B is column major
        for (int ndx = 0; ndx < aWidth; ndx++) {
            lhs = *(sharedA + row*aWidth + ndx);
            rhs = *(sharedB + col*aWidth + ndx);
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
    int aHeight = 4;    //num of rows in A
    const int aWidth = 4;     //num of cols in A
    const int bHeight = 4;    //num of rows in B - this must be the same as aWidth for AB to work
    int bWidth = 4;     //num of cols in B
    double *dev_matrixA, *dev_matrixB, *dev_matrixOut, *dev_sharedA, *dev_sharedB;
    cudaEvent_t start, stop;
    float milliseconds; //how long did we take to do things?

    //bHeight = aWidth;   //Let's just make sure

    //allocate space
    double* matrixA = (double * )malloc(sizeof (double) * aHeight * aWidth);
    double* matrixB = (double * )malloc(sizeof (double) * bHeight * bWidth);        //The operand matrices
    double* matrixOut = (double * )malloc(sizeof (double) * aHeight * bWidth);      //The result matrix
    double* sharedA = (double * )malloc(sizeof (double) * 1024);      //The result matrix
    double* sharedB = (double * )malloc(sizeof (double) * 1024);      //The result matrix


    //fill operands
    fillMatrix(matrixA, aHeight * aWidth);
    fillMatrix(matrixB, bHeight * bWidth);

    //setup memory on device
    cudaMalloc((void**)&dev_matrixA, (aHeight * aWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixB, (bHeight * bWidth) * sizeof(double));
    cudaMalloc((void**)&dev_matrixOut, (aHeight * bWidth) * sizeof(double));
    cudaMalloc((void**)&dev_sharedA, (1024) * sizeof(double));
    cudaMalloc((void**)&dev_sharedB, (1024) * sizeof(double));

    // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(dev_matrixA, matrixA, aHeight * aWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixB, matrixB, bHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixOut, matrixOut, aHeight * bWidth * sizeof(double), cudaMemcpyHostToDevice);

    //Set up problem space dimensions
    //dim3 threadsPerBlock (bWidth, aHeight);
    dim3 threadsPerBlock (2, 2);
    dim3 blocks (2, 2);
    //start timer event
    cudaEventRecord(start);
    //call kernel
    //matrixMultiply<<<blocks,threadsPerBlock>>>(dev_matrixA, dev_matrixB, dev_matrixOut, aHeight, aWidth, bWidth);
    sharedMatrixMultiply<<<blocks,threadsPerBlock>>>(dev_matrixA, dev_matrixB, dev_matrixOut, aHeight, aWidth, bWidth, dev_sharedA, dev_sharedB);
    //stop timer event
    cudaEventRecord(stop);

    //get result from device
    cudaMemcpy(matrixOut, dev_matrixOut, aHeight * bWidth * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sharedA, dev_sharedA, 16 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sharedB, dev_sharedB,  16 * sizeof(double), cudaMemcpyDeviceToHost);
     
    //calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    //free memory
    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixOut);
    cudaFree(sharedA);
    cudaFree(sharedB);

    //Test our calculation
    printMatrixRowMaj(matrixA, aHeight, aWidth);
    printMatrixColMaj(matrixB, bHeight, bWidth);
    printMatrixRowMaj(matrixOut, aHeight, bWidth);
    printMatrixRowMaj(sharedA, 4, 4);
    printMatrixColMaj(sharedB, 4, 4);



    return 0;
}
