#define pool_width 14 
#define pool_width1 12
#define pool_output 10
#define pool_output4 18
#define sr_size 6 // 18*3
#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
typedef struct  mask_struct {
           float dat[3][3];

}mask_struct;

typedef struct  inp_struct {
           float16 dat;

}inp_struct;

typedef struct  sum_struct {
          float running_sum_0 ;
           float running_sum_1 ;
           float running_sum_2;
           float running_sum_3;
           float running_sum_4;
           float running_sum_5;  
           float running_sum_6;
           float running_sum_7;
           float running_sum_8;
           float running_sum_9;
           float running_sum_10;
           float running_sum_11;
           float running_sum_12;
           float running_sum_13;

}sum_struct;

channel inp_struct c0 __attribute__((depth(700))) ;
channel mask_struct c1;
channel int c2;
channel sum_struct c3 __attribute__((depth(700))) ;
channel sum_struct c7 __attribute__((depth(700))) ;
channel int c4;
channel int c5;
channel int c6;
channel int c8;
__kernel void memrd (__global float  * const restrict input, __global float * const restrict mask,const int inputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int input_height, const int window)
{
mask_struct mask_loc;
inp_struct inp;
int d=0;
int index = 0;
int z=0,k=0;
int inputWidth1 = inputWidth-2; 
int iter = input_height* window * window;
write_channel_intel(  c5, iter);
write_channel_intel( c6, kernel_width);
while(index != iter)
{
   index ++;
 
  for(int r=0;r< kernel_width; r++)
     {
        
            #pragma unroll
   for(int i=0;i<3;i++)
    {
      #pragma unroll
       for(int j=0;j<3;j++)
        {
          mask_loc.dat[i][j] = mask[(d * kernel_width + r ) * 9 + i *3 + j];
        }
       }
         write_channel_intel(c1, mask_loc);
      int t= (z* inputWidth + k) * pool_width  + r * inputWidth * inputWidth;
      int t1=t;
     for(int y=0;y< 19; y++)
        {
           t1= t+y* inputWidth;
         inp.dat = (float16)( input[ t1 +  0],  input[ t1 +  1],  input[ t1 +  2],  input[ t1  + 3],  input[  t1 + 4],  input[ t1 + 5],  input[ t1 + 6],  input[ t1 +  7], input[ t1 + 8],  input[t1 + 9],  input[ t1 +  10],  input[t1 + 11],  input[ t1 + 12],  input[t1 + 13],  input[ t1 + 14],  input[ t1 + 15]);
                 write_channel_intel( c0,  inp);
             
        }   
  
 
       }
int e = d * inputWidth1 * inputWidth1 + (( z * inputWidth1 + k) * pool_width );
  write_channel_intel(c2,e);

if( z== window-1)
 {
     k++;
     z=0;
  }
else
{
z++;
}
if(k== window)
{
     d++;
    k=0;
} 
 
} 
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void conv()
{
mask_struct mask_loc1;
inp_struct inp1[sr_size];
int d=0;
int index = 0;
int z=0,k=0;
sum_struct sum;
    // init shift register values
int iter= read_channel_intel(c5);
int kernel_width = read_channel_intel(c6);
#pragma ivdep
while(index != iter)
{ 
    index++;

    int e1 = read_channel_intel(c2);
  #pragma unroll
  for(int i=0;i<sr_size;i++)
   {
        inp1[i].dat= 0.0f;
    }

for(int r=0;r< kernel_width;r++)
{
  
          mask_loc1 = read_channel_intel(c1);
       
       

      for(int y=0; y<19; y++)
        {
            sum.running_sum_0 =0.0f;
            sum.running_sum_1 = 0.0f;
            sum.running_sum_2 =0.0f;
            sum.running_sum_3 = 0.0f;
            sum.running_sum_4 =0.0f;
            sum.running_sum_5 = 0.0f;  
            sum.running_sum_6 =0.0f;
            sum.running_sum_7 = 0.0f;
            sum.running_sum_8 =0.0f;
            sum.running_sum_9 = 0.0f;
            sum.running_sum_10 =0.0f;
            sum.running_sum_11 = 0.0f;
            sum.running_sum_12 =0.0f;
            sum.running_sum_13 = 0.0f;
          
           #pragma unroll
            for(int i=0;i<sr_size-1;i++)
             {
                  inp1[i]= inp1[i+1];
               }
              
               inp1[sr_size-1] = read_channel_intel(c0);
        
                  #pragma unroll
                    for(int i=0;i<3;i++)
                     {
                      #pragma unroll
                        for(int j=0;j<3;j++)
                        {
                           sum.running_sum_0 += inp1[i].dat[j] * mask_loc1.dat[i][j];
                           sum.running_sum_1 += inp1[i].dat[j+1]* mask_loc1.dat[i][j];
                           sum.running_sum_2 += inp1[i].dat[j+2] * mask_loc1.dat[i][j];
                           sum.running_sum_3 += inp1[i].dat[j+3] * mask_loc1.dat[i][j];
                           sum.running_sum_4 += inp1[i].dat[j+4] * mask_loc1.dat[i][j];
                           sum.running_sum_5 += inp1[i].dat[j+5] * mask_loc1.dat[i][j];
                           sum.running_sum_6 += inp1[i].dat[j+6]* mask_loc1.dat[i][j];
                           sum.running_sum_7 += inp1[i].dat[j+7]* mask_loc1.dat[i][j]; 
                           sum.running_sum_8 += inp1[i].dat[j+8] * mask_loc1.dat[i][j];
                           sum.running_sum_9 += inp1[i].dat[j+9] * mask_loc1.dat[i][j];
                           sum.running_sum_10 += inp1[i].dat[j+10] * mask_loc1.dat[i][j];
                           sum.running_sum_11 += inp1[i].dat[j+11]* mask_loc1.dat[i][j];
                           sum.running_sum_12 += inp1[i].dat[j+12]* mask_loc1.dat[i][j];
                           sum.running_sum_13 += inp1[i].dat[j+13]* mask_loc1.dat[i][j]; 
                             
                           
                       }
                      }         
                      write_channel_intel(c3, sum);
           // if( r==0 && y==2 && e1==0)
           {
             //  for(int i=0;i<sr_size;i++)
                 {
               //    for(int j=0;j<16;j++)
                    {
                 //     printf(" value of input is %f \n", inp1[i].dat[j]);
            }
           }          
          } 
}
}


write_channel_intel(c4,e1);

}
}


__kernel void memwrite(__global float  * const restrict input, __global float * const restrict mask,const int inputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int input_height, const int window)
{
float __attribute__((register)) sum[196];
float __attribute__((register)) sum1[196];
sum_struct sum3;
int d=0;
int index = 0;
int z=0,k=0;
sum_struct sum2;
int inputWidth1 = inputWidth-2; 
while(index !=input_height * window * window)
{ 
   index++;
  
    #pragma unroll
   for(int j=0;j< 196;j++)
    {
       sum[j]= 0.0f; 
     }
  
   #pragma unroll
   for(int j=0;j< 196;j++)
    {
       sum1[j]= 0.0f; 
     }
 
     for(int r=0;r<kernel_width; r++)
       {
         for(int y=0;y<19;y++)
          { 
             sum2 = read_channel_intel(c3);
              #pragma unroll
               for(int j=0;j <182; j++)
                {
                  sum1[j] = sum1[j+14];
                  }
     
             sum1[195] =sum2.running_sum_13;
             sum1[194] =sum2.running_sum_12;
             sum1[193] =sum2.running_sum_11;
             sum1[192] =sum2.running_sum_10;
             sum1[191] =sum2.running_sum_9;
             sum1[190] =sum2.running_sum_8;
             sum1[189] =sum2.running_sum_7;
             sum1[188] =sum2.running_sum_6;
             sum1[187] =sum2.running_sum_5;
             sum1[186] =sum2.running_sum_4;
             sum1[185] =sum2.running_sum_3;
             sum1[184] =sum2.running_sum_2;
             sum1[183] =sum2.running_sum_1;
             sum1[182] =sum2.running_sum_0;
                   
           }
        #pragma unroll
      for(int i=0;i< 196;i++)
        {
         sum[i]+= sum1[i];
        }
   
         }
    
            int e1 = read_channel_intel(c4);
            int e2=e1; 
            int e3 =0;
             
             
             for(int y=0; y< pool_width;y++)    	    
            { 
              write_channel_intel(c8,e2);
             sum3.running_sum_0  = sum[e3 + 0];
             sum3.running_sum_1 = sum[e3+1];
             sum3.running_sum_2 = sum[e3 +2];
             sum3.running_sum_3 = sum[e3 + 3];
             sum3.running_sum_4 = sum[e3 +4];
             sum3.running_sum_5 = sum[e3 +5];
             sum3.running_sum_6 = sum[e3 +6];
             sum3.running_sum_7 = sum[e3 +7]; 
             sum3.running_sum_8 = sum[e3 +8];
             sum3.running_sum_9 = sum[e3 +9];
             sum3.running_sum_10 = sum[e3 +10];
             sum3.running_sum_11= sum[e3 +11];
             sum3.running_sum_12 = sum[e3 +12]; 
             sum3.running_sum_13 = sum[e3+13];
             write_channel_intel(c7, sum3);
              e3 = e3+pool_width;                
              e2 = e2+ inputWidth1;  
            }                 
              

}

}

__kernel void outwrite(__global float  * const restrict input, __global float * const restrict mask,const int inputWidth, const int maskwidth,__global float * const restrict output, const int kernel_width, const int window2, const int input_height, const int window)
{
  int index = 0;
  sum_struct sum;
    while(index != input_height * window * window * 14)
   {
  index++;
   sum = read_channel_intel(c7);
  int e1 = read_channel_intel(c8);
  output[e1+0] = sum.running_sum_0;
  output[e1+1] = sum.running_sum_1;
  output[e1+2] = sum.running_sum_2;
  output[e1+3] = sum.running_sum_3;
  output[e1+4] = sum.running_sum_4;
  output[e1+5] = sum.running_sum_5;
  output[e1+6] = sum.running_sum_6;
  output[e1+7] = sum.running_sum_7;
  output[e1+8] = sum.running_sum_8;
  output[e1+9] = sum.running_sum_9;
  output[e1+10] = sum.running_sum_10;
  output[e1+11] = sum.running_sum_11;
  output[e1+12] = sum.running_sum_12;
  output[e1+13] = sum.running_sum_13;
  }

} 
