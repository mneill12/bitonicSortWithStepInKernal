
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda_runtime_api.h>

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>


//Deinfe Size ints
int size;
int blockSize;
int threadCount;
int phases;
const int randMax = 100;

//Function headers 
int* createUnsortedArray();
void printTime(clock_t start, clock_t stop);
void bitonicSetup(int*);
void bitonicSort();
bool ispowerOfTwo(int n);
int getBlockCount();
int getThreadCount();
int getElementCount();
int* getComparasonElements();
bool checkCorrectness(int* elements);
void printElements(int* elements);

struct ComparisonElements {
	int element1;
	int element2;
};
clock_t begin, end;

struct DirectionalComparisonElements{

	int* directionalComparisonElements;
	bool direction;
};

int* createUnsortedArray(){

	//Get size and cuda dimentions from user input
	size = getElementCount();
	blockSize = getBlockCount();
	threadCount = getThreadCount();

	int* elements = (int*)malloc(size*sizeof(int));

	for (int i = 0; i < size; ++i){
		elements[i] = rand() % randMax - rand() % 5;
	}

	return elements;

}

bool isPowerOfTwo(int n ){

	double logBase2 = log2(double(n));

	if (round(logBase2) == logBase2)
	{
		return true;
	}

	else
	{
		return false;
	}
}

int getElementCount(){

	int inputCount;

	bool powerOfTwo = false;
	while (!powerOfTwo){

		printf("Enter amount of elements to be sorted: ");
		scanf_s("%d", &inputCount);
		printf("You entered: %d\n", inputCount);

		if (isPowerOfTwo(inputCount))
		{
			powerOfTwo = true;
		}
		else
		{
			printf("/nNot a power of 2, please re enter/n");
		}
	}

	return inputCount;
}

int getThreadCount(){

	int inputCount;

	bool powerOfTwo = false;
	while (!powerOfTwo){

		printf("Enter amount of threads per block: ");
		scanf_s("%d", &inputCount);
		printf("You entered: %d\n", inputCount);

		if (isPowerOfTwo(inputCount))
		{
			powerOfTwo = true;
		}
		else
		{
			printf("/nNot a power of 2, please re enter/n");
		}
	}

	return inputCount;
}

int getBlockCount(){

	int inputCount;

	bool powerOfTwo = false;
	while (!powerOfTwo){

		printf("Enter amount of blocks: ");
		scanf_s("%d", &inputCount);
		printf("You entered: %d\n", inputCount);

		if (isPowerOfTwo(inputCount))
		{
			powerOfTwo = true;
		}
		else
		{
			printf("/nNot a power of 2, please re enter/n");
		}
	}
	return inputCount;
}

bool checkCorrectness(int* elements){

	bool inOrder = true;

	for (int i = 1; i < size; i ++ ){
		
		if (elements[i] < elements[i - 1]){
			inOrder = false;
		}

		return inOrder;
	}
}

void printElements(int* elements){

	for (int i = 0; i < size; i++){
		printf(" %d 'th element: %d \n", i, elements[i]);
	}
}

void printTimeTaken(clock_t begin, clock_t end)
{
	double timeTaken = ((double)(end - begin)) / CLOCKS_PER_SEC;
	printf("Time taken: %.3fs\n", timeTaken);
}


__global__ void bitonicSort(int* elements, int subSequenceSize, int steps){


	//1printf("Kernal Called!!!!");
	/*
	Here we get our first thread var i and j.
	we get j by knowing the size of the subsequence and then halfing it, this gives us the rang that values should be comapired for this step.
	As we go down the steps, we'll be halfing j until step = 1;
	*/

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	//	int rangeOfComparison = (subSequenceSize / 2);
	//	int j = i + rangeOfComparison;

	//if ((j < size) && (i % subSequenceSize == 0)){

	

	int rangeOfComparison = (subSequenceSize / 2);
	for (int step = steps; step >= 1; step--){

		//This xor op checks that our second value is bigger than our i value
		if ((i ^ rangeOfComparison) > i){

			//assending
			if ((i / subSequenceSize) % 2 == 0){
				//printf("In Assending i^x %d \n", elements[i ^ rangeOfComparison]);
			
			//	printf("Element i: %d, element I^x %d \n",elements[i], elements[i ^ rangeOfComparison]);
			

				if (elements[i] > elements[i ^ rangeOfComparison]) {
					int temp = elements[i];
					elements[i] = elements[i ^ rangeOfComparison];
					elements[i ^ rangeOfComparison] = temp;
				}

			}
			else{
				//printf("In desending i^x %d \n", elements[i ^ rangeOfComparison]);
				if (elements[i] < elements[i ^ rangeOfComparison]) {
					//printf(" Dessending i: %d, Range of Comparion: %d , i / rangeOfComparison %d comparison Value %d \n", i, rangeOfComparison, i / subSequenceSize, i ^ rangeOfComparison);
					int temp = elements[i];
					elements[i] = elements[i ^ rangeOfComparison];
					elements[i ^ rangeOfComparison] = temp;
				}

			}

		}
		__syncthreads();

		rangeOfComparison = rangeOfComparison / 2;
	}
}

void bitonicSetup(int* elements ){

	int* d_elements;

	//get "phases" so we know how many times we need to send array over to device  
	phases = int(log2(double(size)));

	//General cuda managment here : Allocate on device, array isn't going to change  in size
	cudaMalloc(&d_elements, size*sizeof(int));
	cudaMemcpy(d_elements, elements, size*sizeof(int), cudaMemcpyHostToDevice);
	dim3 blocks(blockSize, 1);    /* Number of blocks   */
	dim3 threads(threadCount, 1);  /* Number of threads  */

	//printf("Phases %d/n", phases);

	for (int currentPhase = 1; currentPhase <= phases; currentPhase++){

		//Get the  size of each sub sequence and the amount of "Steps" in the individual sub sequences 
		int subSequenceSize = int(pow(double(2), double(currentPhase)));

		int steps = int(log2((double)subSequenceSize));


		printf("\n");
		bitonicSort<<<blockSize, threadCount >>>(d_elements, subSequenceSize, steps);

	}

	cudaMemcpy(elements, d_elements, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_elements);
}

int main(void){

	int* elements = createUnsortedArray();

	
	//Print elements before and after sort
	//printElements(elements);
	begin = clock();
	bitonicSetup(elements);
	end = clock();
	//printElements(elements);

	if (checkCorrectness){
		printf("Elements in order ");
	}

	else{
		printf("Elements out of order");
	}

	printTimeTaken(begin, end);
	free(elements);

}



/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/