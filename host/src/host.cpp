// faster_rcnn.cpp
//

#define AOCL_ALIGNMENT 64
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<cstddef>
#include <time.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <stdlib.h>
//#include "data.h"
//#include "data_res2_0_branch.h"
//#include "data_res2_0_branch2a.h"
//#include "data_res2_1_branch.h"
//#include "data_res2_2_branch.h"
//#include "data_res3_0_branch.h"
//#include "data_res3_1_branch.h"
//#include "data_res3_2_branch.h"
//#include "data_res3_3_branch.h"
//#include "data_res4_0_branch.h"
//#include "data_res4_1_branch.h"
//#include "data_res4_0_1branch.h"
//#include "data_res4_2_branch.h"
//#include "data_res4_3_branch.h"
//#include "data_res4_4_branch.h"
//#include "data_res4_5_branch.h"
//#include "data_convrpn_branch.h"
//#include "data_rpn_cls.h"
//#include "data_generate_branch.h"
//#include "data_res5_0_branch.h"
//#include "data_res5_1_branch.h"
//#include "data_res5_2_branch.h"
//#include "data_cls_score.h"
//#include "data_bbox_pred.h"
#include "data1.h"


using namespace aocl_utils;


#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif
////inputs //////////////
// Constants
//const unsigned int inputSignalWidth  = 8;
//const unsigned int inputSignalHeight = 8;
bool init_opencl();

//cl_uint inputSignal[inputSignalWidth][inputSignalHeight] =
//{
//	{3, 1, 1, 4, 8, 2, 1, 3},
//	{4, 2, 1, 1, 2, 1, 2, 3},
//	{4, 4, 4, 4, 3, 2, 2, 2},
//	{9, 8, 3, 8, 9, 0, 0, 0},
//	{9, 3, 3, 9, 0, 0, 0, 0},
//	{0, 9, 0, 8, 0, 0, 0, 0},
//	{3, 0, 8, 8, 9, 4, 4, 4},
//	{5, 9, 8, 1, 8, 1, 1, 1}
//};

//const unsigned int outputSignalWidth  = 6;
//const unsigned int outputSignalHeight = 6;

//cl_uint outputSignal[outputSignalWidth][outputSignalHeight];

//const unsigned int maskWidth  = 3;
//const unsigned int maskHeight = 3;
#define STRING_BUFFER_LEN 1024

//cl_uint mask[maskWidth][maskHeight] =
//{
//	{1, 1, 1}, {1, 0, 1}, {1, 1, 1},
//};
//added new to find platform
cl_platform_id platform = NULL;
cl_platform_id* platforms;
cl_uint numPlatforms;
static void display_device_info( cl_device_id device );
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
/// inputs ///////////////
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
void cleanup();

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{          
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

///
//	
int main(int argc, char** argv)
{   /// added to read data file 
//extern CL_API_ENTRY cl_int CL_API_CALL;
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue1;
        cl_command_queue queue2;
        cl_command_queue queue3;  
	cl_program program;
	cl_kernel kernel1;
        cl_kernel kernel2;
        cl_kernel kernel3;  
	cl_mem inputSignalBuffer_r;
        cl_mem inputSignalBuffer_g;
        cl_mem inputSignalBuffer_b;
        cl_mem pool1Buffer;
        cl_mem res2_0_branch1_bnBuffer;
	cl_mem maskBuffer_r;
        cl_mem maskBuffer_g;
        cl_mem maskBuffer_b;
        cl_mem biasBuffer;
        cl_mem res2_0_branch1_w_Buffer;///names tells about the variable name
        cl_mem res2_0_branch1_bn_b_Buffer;
        cl_mem res2_0_branch2a_w_Buffer;
        cl_mem res2_0_branch2a_bn_b_Buffer;
        cl_mem res2_0_branch2b_bn_b_Buffer;
        cl_mem res2_0_branch2b_w_Buffer;
        cl_mem res2_0_branch2b_Buffer;
        cl_mem res2_0_branch2a_Buffer;
        cl_mem res2_0_branch2c_bn_Buffer;
        cl_mem res2_0_branch2c_bn_b_Buffer;
        cl_mem res2_0_branch2c_w_Buffer;
        cl_mem res2_1_branch2a_w_Buffer;
        cl_mem res2_1_branch2a_Buffer;
        cl_mem res2_1_branch2a_bn_b_Buffer;
        cl_mem res2_1_branch2b_Buffer;
        cl_mem res2_1_branch2b_bn_b_Buffer;
        cl_mem res2_1_branch2b_w_Buffer;
        cl_mem res2_1_branch2c_bn_b_Buffer;
        cl_mem res2_1_branch2c_w_Buffer;
        cl_mem res2_1_branch2c_bn_Buffer;
        cl_mem res2_2_branch2a_w_Buffer;
        cl_mem res2_2_branch2a_bn_b_Buffer;
        cl_mem res2_2_branch2a_Buffer;
        cl_mem res2_2_branch2b_Buffer;
        cl_mem res2_2_branch2b_bn_b_Buffer;
        cl_mem res2_2_branch2b_w_Buffer;
        cl_mem res2_2_branch2c_bn_b_Buffer;
        cl_mem res2_2_branch2c_w_Buffer;
        cl_mem res2_2_sum_Buffer; /// no need of res2_2_branch2c_bn here
        cl_mem res3_0_branch2a_w_Buffer;
        cl_mem res3_0_branch2a_bn_b_Buffer;
        cl_mem res3_0_branch1_w_Buffer;
        cl_mem res3_0_branch1_bn_b_Buffer;
        cl_mem res3_0_branch2a_Buffer;
        cl_mem res3_0_branch1_bn_Buffer;
        cl_mem res3_0_branch2b_bn_b_Buffer;
        cl_mem res3_0_branch2b_w_Buffer;
        cl_mem res3_0_branch2c_bn_b_Buffer;
        cl_mem res3_0_branch2c_w_Buffer;
        cl_mem res3_0_branch2c_bn_Buffer;
        cl_mem res3_0_branch2b_Buffer;
        cl_mem res3_1_branch2a_w_Buffer;
        cl_mem res3_1_branch2a_bn_b_Buffer;
        cl_mem res3_1_branch2a_Buffer;
        cl_mem res3_1_branch2b_w_Buffer;
        cl_mem res3_1_branch2b_Buffer;
        cl_mem res3_1_branch2b_bn_b_Buffer;
        cl_mem res3_1_branch2c_bn_b_Buffer;
        cl_mem res3_1_branch2c_w_Buffer;
        cl_mem res3_1_branch2c_bn_Buffer;
        cl_mem res3_2_branch2a_w_Buffer;
        cl_mem res3_2_branch2a_bn_b_Buffer;
        cl_mem res3_2_branch2a_Buffer;
        cl_mem res3_2_branch2b_bn_b_Buffer;
        cl_mem res3_2_branch2b_w_Buffer;
        cl_mem res3_2_branch2b_Buffer;
        cl_mem res3_2_branch2c_bn_b_Buffer;
        cl_mem res3_2_branch2c_w_Buffer;
        cl_mem res3_2_branch2c_bn_Buffer;
        cl_mem res3_3_branch2a_w_Buffer;
        cl_mem res3_3_branch2a_bn_b_Buffer;
        cl_mem res3_3_branch2a_Buffer;
        cl_mem res3_3_branch2b_bn_b_Buffer;
        cl_mem res3_3_branch2b_w_Buffer;
        cl_mem res3_3_branch2b_Buffer;
        cl_mem res3_3_branch2c_w_Buffer;
        cl_mem res3_3_branch2c_bn_b_Buffer;
        cl_mem res3_3_sum_Buffer; //// no need of res3_3_branch2c_bn here//
        cl_mem res3_4_branch2a_Buffer;
        cl_mem res4_0_branch2a_w_Buffer;
        cl_mem res4_0_branch2a_bn_b_Buffer;
        cl_mem res4_0_branch2a_Buffer;
        cl_mem res4_0_branch1_w_Buffer;
        cl_mem res4_0_branch2b_bn_b_Buffer;
        cl_mem res4_0_branch2b_Buffer;
        cl_mem res4_0_branch1_bn_b_Buffer;
        cl_mem res4_0_branch2b_w_Buffer;
        cl_mem res4_0_branch2c_bn_b_Buffer;
        cl_mem res4_0_branch2c_w_Buffer;
        cl_mem res4_0_branch2c_bn_Buffer;
        cl_mem res4_1_branch2a_w_Buffer;
        cl_mem res4_1_branch2a_Buffer;
        cl_mem res4_1_branch2a_bn_b_Buffer;
        cl_mem res4_1_branch2b_bn_b_Buffer;
        cl_mem res4_1_branch2b_w_Buffer;
        cl_mem res4_1_branch2b_Buffer;
        cl_mem res4_1_branch2c_bn_b_Buffer;
        cl_mem res4_1_branch2c_w_Buffer;
        cl_mem res4_1_branch2c_bn_Buffer;
        cl_mem res4_2_branch2a_w_Buffer;
        cl_mem res4_2_branch2a_Buffer;
        cl_mem res4_2_branch2a_bn_b_Buffer;
        cl_mem res4_2_branch2b_w_Buffer;
        cl_mem res4_2_branch2b_Buffer;
        cl_mem res4_2_branch2b_bn_b_Buffer;
        cl_mem res4_2_branch2c_bn_b_Buffer;
        cl_mem res4_2_branch2c_w_Buffer;
        cl_mem res4_2_branch2c_bn_Buffer;
        cl_mem res4_3_branch2a_w_Buffer;
        cl_mem res4_3_branch2a_Buffer;
        cl_mem res4_3_branch2a_bn_b_Buffer;
        cl_mem res4_3_branch2b_w_Buffer;
        cl_mem res4_3_branch2b_Buffer;
        cl_mem res4_3_branch2b_bn_b_Buffer;
        cl_mem res4_3_branch2c_bn_b_Buffer;
        cl_mem res4_3_branch2c_w_Buffer;
        cl_mem res4_3_branch2c_bn_Buffer;
        cl_mem res4_4_branch2a_w_Buffer;
        cl_mem res4_4_branch2a_bn_b_Buffer;
        cl_mem res4_4_branch2a_Buffer;
        cl_mem res4_4_branch2b_bn_b_Buffer;
        cl_mem res4_4_branch2b_w_Buffer;
        cl_mem res4_4_branch2c_bn_b_Buffer;
        cl_mem res4_4_branch2b_Buffer;
        cl_mem res4_4_branch2c_bn_Buffer;
        cl_mem res4_4_branch2c_w_Buffer;
        cl_mem res4_5_branch2a_w_Buffer;
        cl_mem res4_5_branch2a_Buffer;   
        cl_mem res4_5_branch2a_bn_b_Buffer;
        cl_mem res4_5_branch2b_w_Buffer;
        cl_mem res4_5_branch2b_bn_b_Buffer;
        cl_mem res4_5_branch2b_Buffer;
        cl_mem res4_5_branch2c_bn_Buffer;
        cl_mem res4_5_branch2c_w_Buffer;
        cl_mem res4_5_branch2c_bn_b_Buffer;
        cl_mem conv_rpn_w_Buffer;
        cl_mem conv_rpn_b_Buffer;
        cl_mem conv_rpn_Buffer;
        cl_mem rpn_cls_logits_b_Buffer;
        cl_mem rpn_cls_logits_w_Buffer;
        cl_mem rpn_cls_probs_Buffer;
        cl_mem rpn_bbox_pred_w_Buffer;
        cl_mem rpn_bbox_pred_b_Buffer;
        cl_mem rpn_bbox_pred_Buffer;
        cl_mem anchor_Buffer;  /// after data preparation from all_anchor, all_anchor is send 
        cl_mem im_info_Buffer;
        cl_mem rpn_rois_Buffer;
        cl_mem pool5_Buffer;
        cl_mem res5_0_branch2a_w_Buffer;
        cl_mem res5_0_branch2a_bn_b_Buffer;
        cl_mem res5_0_branch2a_Buffer;
        cl_mem res5_0_branch2b_w_Buffer;
        cl_mem res5_0_branch2b_bn_b_Buffer;
        cl_mem res5_0_branch2b_Buffer;
        cl_mem res5_0_branch2c_bn_Buffer;
        cl_mem res5_0_branch2c_bn_b_Buffer;
        cl_mem res5_0_branch2c_w_Buffer;
        cl_mem res5_0_branch1_w_Buffer;
        cl_mem res5_0_branch1_bn_b_Buffer;
        cl_mem res5_0_branch1_bn_Buffer;
        cl_mem res5_1_branch2a_w_Buffer;
        cl_mem res5_1_branch2a_bn_b_Buffer;
        cl_mem res5_1_branch2a_Buffer;
        cl_mem res5_1_branch2b_w_Buffer;
        cl_mem res5_1_branch2b_bn_b_Buffer;
        cl_mem res5_1_branch2b_Buffer;
        cl_mem res5_1_branch2c_w_Buffer;
        cl_mem res5_1_branch2c_bn_b_Buffer;
        cl_mem res5_1_branch2c_bn_Buffer;
        cl_mem res5_2_branch2a_Buffer;
        cl_mem res5_2_branch2a_w_Buffer;
        cl_mem res5_2_branch2a_bn_b_Buffer;
        cl_mem res5_2_branch2b_Buffer;
        cl_mem res5_2_branch2b_w_Buffer;
        cl_mem res5_2_branch2b_bn_b_Buffer;
        cl_mem res5_2_branch2c_w_Buffer;
        cl_mem res5_2_branch2c_bn_b_Buffer;
        cl_mem res5_2_branch2c_bn_Buffer;
        cl_mem res5_pool_Buffer;
        cl_mem cls_score_w_Buffer;
        cl_mem cls_score_b_Buffer;
        cl_mem cls_score_Buffer;
        cl_mem bbox_pred_w_Buffer;
        cl_mem bbox_pred_b_Buffer;
        cl_mem bbox_pred_Buffer;
        cl_mem pred_bbox_Buffer;
        cl_mem cls_prob_Buffer;
        cl_mem output_Buffer;
        cl_mem bias_Buffer;
       //float res5_pool[kernel_number_8_2][pool8_height/7][pool8_width/7];
       //float pred_bbox[pool_height10][pool_width10];
       //float cls_prob[pool_height9][pool_width9];
           //   float res3_4_branch2a[kernel_number_4_1][pool2_height][pool1_width];
//void *inputSignal_r = NULL;
//posix_memalign ( &inputSignal_r, AOCL_ALIGNMENT, input_width * inputSignalWidth * inputSignalWidth);
//for(int w=0;w< height; w++)
{
for(int i=0;i< input_width;i++)
 {
  for (j=0;j<inputSignalHeight;j++)
  {
    for(r=0;r<inputSignalWidth;r++)
     {
       inputSignal_r[i][j][r]= random()%10;
        }
   }
}
}
  printf("data r recieved is \n");
//for(int w=0;w< height;w++)
{
 for(int i=0;i< input_width;i++)
 {
  for(r=0;r<inputSignalHeight;r++)
  {
    //printf("[");
   for(j=0;j<inputSignalWidth;j++)
     {
 //    printf(" %f ,", inputSignal_r[i][r][j]);
     }
  // printf(" \n  \n");
   }
//printf("\n, \n");
}
//printf("\n, \n");
}
printf("data g recieved is \n");
  
for (int w=0;w< input_height;w++)
{
  for(int i=0;i<input_width;i++)
   { 
    for(int r=0;r< maskHeight;r++)
     {
      for(int j=0;j<maskWidth;j++)
       {
mask_r[w][i][r][j]= random()%4;
        }
       }
     }
    }
       
//for(r=3;r<inputSignalHeight-3;r++)
  //{
    //printf("[");
   //for(j=3;j<inputSignalWidth-3;j++)
     //{
       //printf(" %f ,", inputSignal_g[r][j]);
     //}
      //printf("],\n");
   //} 
printf("data b recieved is \n");
 
//for(r=3;r<inputSignalHeight-3;r++)
  //{
    // printf("[");
   //for(j=3;j<inputSignalWidth-3;j++)
     //{
      // printf(" %f ,", inputSignal_b[r][j]);
     //}
     //printf("],\n");
   //} 
//anchor data preparation for  generate proposals algorithm///
//since it is one time thing do not need to be done for multiple times ///
//float shift_x[pool_output5*pool_output5], shift_y[pool_output5*pool_output5],shifts[pool_output5* pool_output5][kernel_number_7_2];
//float all_anchors[pool_output5*pool_output5][kernel_number_7_1][kernel_number_7_2];
//float feat_stride= 16.0; //determined by the weight
//for (int i=0; i< pool_output5;i++)
//{
   //for (int j=0;j<pool_output5;j++)
  //{
    //shift_x[i* pool_output5+ j]= j * feat_stride;
    //shift_y[i* pool_output5+ j]= i * feat_stride;
//}
//}

//for(int j=0;j<pool_output5* pool_output5;j++)
  //  {
    //  shifts[j][0]= shift_x[j];
     // shifts[j][1]= shift_y[j];
      //shifts[j][2]= shift_x[j];
      //shifts[j][3]= shift_y[j];
//}
//for(int i=0;i<pool_output5* pool_output5;i++)
//{
 // for(int j=0;j<kernel_number_7_1;j++)  
   //{ 
     //for(int k=0;k<kernel_number_7_2;k++)
    //{
      // all_anchors[i][j][k]=shifts[i][k]+ anchor[j][k];
      //}
   //}
//}
//printf(" all anchors value is \n");
//for(int i=0;i<pool_output5 * pool_output5;i++)
//{ 
  //for(int j=0;j<kernel_number_7_1;j++)
  // {
    //for(int k=0;k<kernel_number_7_2;k++)
     //{

   //   printf("  %f  ", all_anchors[i][j][k]);
    //} 
    //  printf(" \n");
//}
//printf("\n");
//}

    // First, select an OpenCL platform to run on.  
     clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms, NULL);
      std::cout << "number of platform is : " << numPlatforms << std::endl;

     if(!init_opencl()) {
                   return -1;
                    }
              

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platforms[i], 
            CL_DEVICE_TYPE_ALL, 
            0,
            NULL,
            &numDevices);
            
            std::cout << "errNum : " << errNum << std::endl;
            std::cout << "cl_sucess : " << CL_SUCCESS << std::endl;
            std::cout << "cl_device is : " << CL_DEVICE_NOT_FOUND << std::endl;
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {           
                        std::cout << "if part is running " << std::endl;  
			checkErr(errNum, "clGetDeviceIDs");
                       
        }
	    else if (numDevices > 0) 
		{        std::cout << "else if part is running " << std::endl;
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platforms[i],
				CL_DEVICE_TYPE_ALL,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}
          std::cout << "DeviceID is : " << &deviceIDs[1] << std::endl; 
         std::cout << "no else if part " << std::endl;
	// Check to see if we found at least one CPU device, otherwise return
	if (deviceIDs == NULL) {
                 std::cout << "if_device id part is running " << std::endl;
		std::cout << "No FPGA device found" << std::endl;
		exit(-1);
	}
      std::cout << "no if part " << std::endl;

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[i],
        0
    };
    std::cout << "context property is  " << contextProperties <<std::endl;
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
     std::cout << "context is  " << context <<std::endl;
	checkErr(errNum, "clCreateContext");
    std::cout << "maybe error is here after checkERR  " <<std::endl;
	
   std::cout << "maybe error is here after another checkERR  " <<std::endl;
  // added to define program ""'
  std::string binary_file = getBoardBinaryFile("conv_gen_shift4", deviceIDs[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), deviceIDs, numDevices);
	// Create program from source
   std::cout << "prgram object is   " <<program<<std::endl;
	// Build program
	errNum = clBuildProgram(program, 0, NULL, "", NULL, NULL);
          std::cout << "maybe error is here after this checkERR  " <<std::endl;
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);
    	
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
	    
             checkErr(errNum, "clBuildProgram");
             std::cout << "maybe error is here before another checkERR  " <<std::endl;
    }

	// Create kernel object for r file
	
	// Now allocate buffers
       
	inputSignalBuffer_r = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * input_width * inputSignalHeight* inputSignalWidth,
		static_cast<void *>(inputSignal_r),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");
     
	maskBuffer_r = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height * input_width* maskHeight * maskWidth,
		static_cast<void *>(mask_r),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
     bias_Buffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) *  input_height,
		static_cast<void *>(bias),
                 &errNum);
        checkErr(errNum, "clCreateBuffer(mask)");
  
output_Buffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY ,
		sizeof(float) * input_height* outputSignalHeight * outputSignalWidth , /// to do padding////
 		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(output)");
cl_event event1;
cl_event event2;
cl_event event3;
cl_ulong time_start;
cl_ulong time_end;
cl_ulong time_start1;
cl_ulong time_end1;
int y=0;
int z=1;
time_start1 = clock();
//for(int i=0;i<2;i++)
//{
kernel1 = clCreateKernel(
		program,
		"memrd",
		&errNum);
	checkErr(errNum, "clCreateKernel");
        queue1 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inputSignalBuffer_r);
errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_uint), &inputSignalWidth);
errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_uint), &maskWidth_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_uint), &kernel_width);
errNum |= clSetKernelArg(kernel1, 6, sizeof(cl_uint), &window_width);
errNum |= clSetKernelArg(kernel1, 7, sizeof(cl_uint), &input_height);
errNum |= clSetKernelArg(kernel1, 8, sizeof(cl_uint), &window);


	// Pick the first device and create command queue.
    const size_t globalWorkSize1[1] = {1};
    const size_t localWorkSize1[1]  = {1};
errNum = clEnqueueNDRangeKernel(
		queue1, 
		kernel1, 
		1, 
		NULL,
        globalWorkSize1, 
		localWorkSize1,
        0, 
		NULL, 
		&event1);
checkErr(errNum, "clEnqueueNDRangeKernel");
   


const size_t globalWorkSize2[1] = {1};
//const size_t localWorkSize1[1]  = {1};

kernel3 = clCreateKernel(
		program,
		"memwrite",
		&errNum);
	checkErr(errNum, "clCreateKernel");
        queue3 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel3, 0, sizeof(cl_mem), &inputSignalBuffer_r);
errNum |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel3, 2, sizeof(cl_uint), &inputSignalWidth);
errNum |= clSetKernelArg(kernel3, 3, sizeof(cl_uint), &maskWidth_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel3, 4, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel3, 5, sizeof(cl_uint), &kernel_width);
errNum |= clSetKernelArg(kernel3, 6, sizeof(cl_uint), &window_width);
errNum |= clSetKernelArg(kernel3, 7, sizeof(cl_uint), &input_height);
errNum |= clSetKernelArg(kernel3, 8, sizeof(cl_uint), &window);
//const size_t globalWorkSize2[1] = {1};
//const size_t localWorkSize1[1]  = {1};
errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel3, 
		1, 
		NULL,
        globalWorkSize2, 
		localWorkSize1,
        0, 
		NULL, 
		&event3);
checkErr(errNum, "clEnqueueNDRangeKernel");


kernel3 = clCreateKernel(
		program,
		"outwrite",
		&errNum);
	checkErr(errNum, "clCreateKernel");
        queue3 = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
  errNum |= clSetKernelArg(kernel3, 0, sizeof(cl_mem), &inputSignalBuffer_r);
errNum |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &maskBuffer_r);
//errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &res5_0_branch2b_bn_b_Buffer);
errNum |= clSetKernelArg(kernel3, 2, sizeof(cl_uint), &inputSignalWidth);
errNum |= clSetKernelArg(kernel3, 3, sizeof(cl_uint), &maskWidth_1);  /// maskwidth_1 acting as window for mask16
errNum |= clSetKernelArg(kernel3, 4, sizeof(cl_mem), &output_Buffer);
errNum |= clSetKernelArg(kernel3, 5, sizeof(cl_uint), &kernel_width);
errNum |= clSetKernelArg(kernel3, 6, sizeof(cl_uint), &window_width);
errNum |= clSetKernelArg(kernel3, 7, sizeof(cl_uint), &input_height);
errNum |= clSetKernelArg(kernel3, 8, sizeof(cl_uint), &window);
//const size_t globalWorkSize2[1] = {1};
//const size_t localWorkSize1[1]  = {1};
errNum = clEnqueueNDRangeKernel(
		queue3, 
		kernel3, 
		1, 
		NULL,
        globalWorkSize2, 
		localWorkSize1,
        0, 
		NULL, 
		&event3);
checkErr(errNum, "clEnqueueNDRangeKernel");

//const size_t globalWorkSize2[1] = {1};
//const size_t localWorkSize1[1]  = {1};


//}

errNum = clEnqueueReadBuffer(
		queue3, 
		output_Buffer, 
		CL_TRUE,
        0, 
		sizeof(float) * input_height*outputSignalHeight * outputSignalWidth, 
		conv,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
time_end1 = clock();

double cpu_time_used= ((double) (time_end1-time_start1))/ CLOCKS_PER_SEC;
 
 //clFinish(queue3);
  //clWaitForEvents(1,&event3);
clGetEventProfilingInfo(event3,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL);
  clGetEventProfilingInfo(event3,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);
    double nanosecs= time_end- time_start;
    //printf(" cpu execution time is %f \n", cpu_time_used);
printf(" true \n");

std::cout<<" output result is   "<<std::endl;

int r=0;
// Output the result buffer//
//for(int w=0;w<1;w++)
{
for (int z=0; z<input_height;z++)
{
for (int y = 0; y < outputSignalHeight; y++)
	{
  	for (int x = 0; x <outputSignalWidth; x++)
		{
                      // std::cout << x+y*(pool_output7)+ z*(pool_output7)*(pool_output7) +w << "        ";  
              std::cout <<conv[z][y][x] << " ";
  
		}
      std::cout << std::endl;
	}
	std::cout << std::endl;
}
//std::cout<< "one ";
}
printf(" execution time is  %0.3f milisecs", nanosecs/1000000.0);
     //cleanup();


	return 0;
}
/////functions added later /////////
// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("INTEL");
  std::cout << std::endl << "platform is " << platform << std::endl;
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
        }
  else {
   printf("Platform: %s\n", getPlatformName(platform).c_str());
   return true;
}
}
//////clean up part added later /////////



 

