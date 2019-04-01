///image data to feed to kernel////
#include<stdio.h>
#include<stdlib.h>
#define pool1_height 12
#define pool1_width 12
const unsigned int inputSignalWidth  = 198; //7 9
const unsigned int inputSignalHeight = 198;
unsigned int r=0;
unsigned int j=0;
const unsigned int window_width =0;
const unsigned int input_width=  64; //32 2048
const unsigned int input_height= 64;
const unsigned int height= 1;
float inputSignal_r[input_width][inputSignalHeight][inputSignalWidth];
const unsigned int outputSignalWidth = 196;
const unsigned int outputSignalHeight= 196;
const unsigned int kernel_width = 64;
const unsigned int ch = 0; // 2 for 3 maskwidth and 0 for 1 maskwidth
const unsigned int kernel_window = kernel_width/16;
float conv [input_height][outputSignalHeight][outputSignalWidth]={0};
const unsigned int input_height1 = input_height;
const unsigned int maskWidth  =3;
const unsigned int maskHeight =3;
const unsigned int maskWidth_1 = input_width/16;
float bias[input_height] = {0};
int window = 14;
float mask_r [input_height][input_width][maskHeight][maskWidth] ;//= {{{{1,1,1}, {1,1,1}, {1,1,1}},{{2,2,2},{2,2,2},{2,2,2}}}};//,{{{2,2,2}, {2,2,2}, {2,2,2}},{{4,4,4},{4,4,4},{4,4,4}}}}; //{{{{1}},{{2}},{{3}},{{4}},{{5}},{{6}},{{7}},{{8}},{{9}},{{10}},{{11}},{{12}},{{13}},{{14}},{{15}},{{16}}}, {{{2}},{{4}},{{6}},{{8}},{{10}},{{12}},{{14}},{{16}},{{18}},{{20}},{{22}},{{24}},{{26}},{{28}},{{30}},{{32}}}}; ////{{{{1}},{{2}},{{3}},{{4}},{{5}},{{6}},{{7}},{{8}},{{9}},{{10}},{{11}},{{12}},{{13}},{{14}},{{15}},{{16}}}, {{{2}},{{4}},{{6}},{{8}},{{10}},{{12}},{{14}},{{16}},{{18}},{{20}},{{22}},{{24}},{{26}},{{28}},{{30}},{{32}}}};//
//= 
//  // {{{{1,1,1}, {1,1,1}, {1,1,1}},{{2,2,2},{2,2,2},{2,2,2}}}, {{{2,2,2}, {2,2,2}, {2,2,2}},{{4,4,4},{4,4,4},{4,4,4}}}} ;////= //  //;// ; //=  ;//=
;//= {{{{1,1,1}}, {{2,2,2}}}, {{{2,2,2}}, {{4,4,4}}}}; 

// to print the input signal


