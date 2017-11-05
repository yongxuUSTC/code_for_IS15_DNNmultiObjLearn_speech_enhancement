#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include "BP_GPU.h"
#include "DevFunc.h"

#define THREUSEMULTIGPU 256

BP_GPU::BP_GPU(int a_GPU_selected, int a_numlayers, int *a_layersizes, int a_bunchsize, float a_lrate, float a_momentum,  
	float a_weightcost,float **weights, float **bias,int a_dropoutflag,float a_visible_omit,float a_hid_omit)
	:GPU_selected(a_GPU_selected),numlayers(a_numlayers),bunchsize(a_bunchsize),momentum(a_momentum),lrate(a_lrate),weightcost(a_weightcost),dropoutflag(a_dropoutflag), visible_omit(a_visible_omit),hid_omit(a_hid_omit)
{
	int i,j;
   int maxlayersize=0;
	//// set GPU num
	cudaGetDeviceCount(&GPU_total);
	printf("Total GPU Device : %d\n",GPU_total);

	if(GPU_selected > GPU_total || GPU_selected < 1)
	{
		printf("GPU Num %d Not In Range %d-%d\n",GPU_selected,1,GPU_total);
		exit(0);
	}
	printf("Use GPU Device : %d\n",GPU_selected);



	int bunch_part[GPU_selected];
	int part = bunchsize/GPU_selected;

	for(i= 0; i< GPU_selected-1;i++)
	{
		bunch_part[i] = part;
	}
	bunch_part[GPU_selected -1] = bunchsize -part*(GPU_selected -1);



	////Init cublas && streams
	dev = (BP_WorkSpace*) malloc(GPU_selected * sizeof(BP_WorkSpace));
	handles = (cublasHandle_t*) malloc(GPU_selected * sizeof(cublasHandle_t)); 	
	streams = (cudaStream_t*) malloc(GPU_selected * sizeof(cudaStream_t));
  gen = (curandGenerator_t*) malloc(GPU_selected * sizeof(curandGenerator_t));

	for(i = 0;i < GPU_selected;i++)
	{
		cudaError_t er;
    curandStatus_t eg;
    
		er = cudaSetDevice(i);
		//er = cudaSetDevice(1);
		if (er!=cudaSuccess)
			printf("cudaSetDevice(%d) failed\n",i);


		er =cudaStreamCreate(&(streams[i]));
		if (er!=cudaSuccess)
			printf("cudaStreamCreate(%d) failed\n",i);

		cublasStatus_t eb = cublasCreate(&handles[i]);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasCreate(%d) failed\n",i);

		eb = cublasSetStream(handles[i],streams[i]);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasSetStream(handles[%d],streams[%d]) failed\n",i,i);
			
	  eg = curandCreateGenerator(&gen[i] ,CURAND_RNG_PSEUDO_DEFAULT);
		if(eg!=CURAND_STATUS_SUCCESS)
            printf("curandCreateGenerator(%d) failed\n",i);

		eg = curandSetStream(gen[i],streams[i]);
		if(eg!=CURAND_STATUS_SUCCESS)
            printf("curandSetStream(%d) failed\n",i);

		srand(unsigned(time(NULL)));
		curandSetPseudoRandomGeneratorSeed(gen[i] ,rand());
	}
	if(GPU_selected >1)
	{


		for(i =0;i< GPU_selected;i++)
		{

			cudaSetDevice(i);
		  // cudaSetDevice(1);
			for(j =0;j< GPU_selected;j++)
			{			 
		        
				if(j != i)
				{
				    int UVA;
				    cudaDeviceCanAccessPeer(&UVA,j,i);
					if(UVA==0)
					{
					    printf("cudaDeviceCanAccessPeer error\n");
						exit(0);
					}
					else
					{
					    printf("cudaDeviceCanAccessPeer between Device %d and Device %d OK\n",j,i);
					    cudaDeviceEnablePeerAccess(j, 0);
					}
					    
				}
					
			}
		}
	}

	//// Alloc device Memory
	for(i =0; i < numlayers;i++)
	{
		layersizes[i] = a_layersizes[i];
		if (maxlayersize<layersizes[i])
			{maxlayersize=layersizes[i];}
	}

	for(j =0;j< GPU_selected;j++)
	{
	    if(j==0)
		{
		    cudaSetDevice(0);
		   // cudaSetDevice(1);
		    devnew_vf("in", 						MAXCACHEFRAME *layersizes[0], &(dev[j].in));
		    devnew_vf("out", 						bunchsize *layersizes[numlayers -1], &(dev[j].out));
		    //devnew_vi("targ", 					MAXCACHEFRAME, &(dev[j].targ));
			devnew_vf("targ", 					MAXCACHEFRAME*layersizes[numlayers -1], &(dev[j].targ));/////////////////////////////////yongxu
         devnew_vf("DevRandVector",			maxlayersize *bunch_part[j],&(dev[j].DevRandVector));
		devnew_vi("DevSeed",BASICSIZE,&(dev[j].DevSeed));
		    for (i = 1; i< numlayers; i++)
		    {
			    devnew_vf("bias", 	 layersizes[i], &(dev[j].bias[i]));
			    devnew_vf("weights", layersizes[i] *layersizes[i-1], &(dev[j].weights[i]));
			    devnew_vf("delta_bias", 	 layersizes[i], &(dev[j].delta_bias[i]));
			    devnew_vf("delta_weights", layersizes[i] *layersizes[i-1], &(dev[j].delta_weights[i]));
			    devnew_vf("layer_y", bunchsize *layersizes[i], &(dev[j].layer_y[i]));
			    devnew_vf("layer_x", bunchsize *layersizes[i], &(dev[j].layer_x[i]));
			    devnew_vf("layer_dedy", bunchsize *layersizes[i], &(dev[j].layer_dedy[i]));
			    devnew_vf("layer_dydx", bunchsize *layersizes[i], &(dev[j].layer_dydx[i]));
			    devnew_vf("layer_dedx", bunchsize *layersizes[i], &(dev[j].layer_dedx[i]));
			    devnew_vf("layer_ydedx", layersizes[i] *layersizes[i-1], &(dev[j].layer_ydedx[i]));
			    devnew_vf("layer_sumdedx", layersizes[i], &(dev[j].layer_sumdedx[i]));
		    }		    
		}
		else
		{
		    cudaSetDevice(j);
		   // cudaSetDevice(1);
		    devnew_vf("in", 						MAXCACHEFRAME *layersizes[0], &(dev[j].in));
		    devnew_vf("out", 						bunch_part[j] *layersizes[numlayers -1], &(dev[j].out));
		    //devnew_vi("targ", 					MAXCACHEFRAME, &(dev[j].targ));
			devnew_vf("targ", 					MAXCACHEFRAME*layersizes[numlayers -1], &(dev[j].targ));////////////////////////////////////yongxu

		    for (i = 1; i< numlayers; i++)
		    {
			    devnew_vf("bias", 	 layersizes[i], &(dev[j].bias[i]));
			    devnew_vf("weights", layersizes[i] *layersizes[i-1], &(dev[j].weights[i]));
			    devnew_vf("delta_bias", 	 layersizes[i], &(dev[j].delta_bias[i]));
			    devnew_vf("delta_weights", layersizes[i] *layersizes[i-1], &(dev[j].delta_weights[i]));
			    devnew_vf("layer_y", bunch_part[j] *layersizes[i], &(dev[j].layer_y[i]));
			    devnew_vf("layer_x", bunch_part[j] *layersizes[i], &(dev[j].layer_x[i]));
			    devnew_vf("layer_dedy", bunch_part[j] *layersizes[i], &(dev[j].layer_dedy[i]));
			    devnew_vf("layer_dydx", bunch_part[j] *layersizes[i], &(dev[j].layer_dydx[i]));
			    devnew_vf("layer_dedx", bunch_part[j] *layersizes[i], &(dev[j].layer_dedx[i]));
			    devnew_vf("layer_ydedx", layersizes[i] *layersizes[i-1], &(dev[j].layer_ydedx[i]));
			    devnew_vf("layer_sumdedx", layersizes[i], &(dev[j].layer_sumdedx[i]));
		    }
		}

	}
	if(GPU_selected >1)
	{
		cudaDeviceSynchronize();
	}

	////copy weights && biases to devices
	for(j =0;j< GPU_selected;j++)
	{

		cudaSetDevice(j);
   //cudaSetDevice(1);
    
		for(i = 1; i< numlayers; i++)
		{
			todev_vf_vf("weights", layersizes[i-1] *layersizes[i], weights[i], dev[j].weights[i], streams[j]);
			todev_vf_vf("bias", layersizes[i], bias[i], dev[j].bias[i], streams[j]);
		}
	}
	if(GPU_selected >1)
	{
		cudaDeviceSynchronize();
	}
	printf("Created net with %d layers, bunchsize %d.\n", numlayers, bunchsize);
}

BP_GPU::~BP_GPU()
{
	int i,j;

	////streams & cublas free	
	for(j =0;j< GPU_selected;j++)
	{

		cudaSetDevice(j);
    //cudaSetDevice(1);

		devfree_vf("in", dev[j].in);
		devfree_vf("out", dev[j].out);
		//devfree_vi("targ", dev[j].targ);
		devfree_vf("targ", dev[j].targ);/////////////////////////////////////////////////yongxu
       devfree_vf("DevRandVector",dev[j].DevRandVector);
	 devfree_vi("DevSeed", dev[j].DevSeed);
		for (i = 1; i< numlayers; i++)
		{
			devfree_vf("weights", dev[j].weights[i]);
			devfree_vf("bias", dev[j].bias[i]);
			devfree_vf("delta_weights", dev[j].delta_weights[i]);
			devfree_vf("delta_bias", dev[j].delta_bias[i]);
			devfree_vf("layer_x", dev[j].layer_x[i]);
			devfree_vf("layer_y", dev[j].layer_y[i]);
			devfree_vf("layer_dedx", dev[j].layer_dedx[i]);
			devfree_vf("layer_dydx", dev[j].layer_dydx[i]);
			devfree_vf("layer_dedy", dev[j].layer_dedy[i]);
			devfree_vf("layer_ydedx", dev[j].layer_ydedx[i]);
			devfree_vf("layer_sumdedx", dev[j].layer_sumdedx[i]);
		}


		cublasDestroy(handles[j]);
		cudaStreamDestroy(streams[j]);
	 curandDestroyGenerator(gen[j]);
	}
	delete[] dev;
}

//void BP_GPU::train(int n_frames, const float* in, const int *targ)
//void BP_GPU::train(int n_frames, const float* in, const float *targ)////////////////////////////by yongxu
void BP_GPU::train(int n_frames, float* in, const float *targ)
{

	int i,j;
	//int t;
	int frames_this_bunch;	// Number of frames to handle this bunch
	int n_input = layersizes[0];
	int out_dims= layersizes[numlayers-1];
	
	float **realin = new float*[GPU_selected];
	//int **realtarg = new int*[GPU_selected];
	float **realtarg = new float*[GPU_selected];///////////////////////////////////by yongxu
	//float *realin;
	//int *realtarg;

	int n_frames_part = n_frames/GPU_selected;

// for (t=0;t<517;t++)
// { printf("in[%d]=%f,",t,in[t]);
//   }
//   printf ("\n");
//   
//   for (t=0;t<200;t++)
//   {printf("targ[%d]=%f,",t,targ[t]);
//   	}
//   printf ("\n");
   
	// First copy data to GPU
	for(i= 0; i< GPU_selected;i++)
	{

		cudaSetDevice(i);
		//cudaSetDevice(1);
		todev_vf_vf("in",n_frames_part * n_input, in + i* n_frames_part* n_input, dev[i].in, streams[i]);
		//todev_vi_vi("targ", n_frames_part, targ + i* n_frames_part, dev[i].targ, streams[i]);
	    todev_vf_vf("targ", n_frames_part * out_dims, targ + i* n_frames_part, dev[i].targ, streams[i]);
	}
	if(GPU_selected >1)
	{
		cudaDeviceSynchronize();
	}
	//printf("Copy Data Sucess , %d Frames\n",n_frames);


	for(i= 0; i< GPU_selected;i++)
	{
		realin[i] = dev[i].in;
		realtarg[i] = dev[i].targ;    
	}

  

	//printf("GPU_selected : %d\n",GPU_selected);
	for (i=0; i< n_frames; i+= bunchsize)
	{
		//printf("i=%d\n",i);
		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;
		if(frames_this_bunch == bunchsize)
		{
			//printf("in \n");
			if(GPU_selected == 1)
				{
					//printf("in-in \n");
					//printf("realin[0][1]=%f,realtarg[0][1]=%f\n",realin[0][1],realtarg[0][1]);//����ط��䲻������Ҳ������
					//printf("dev[0].in[1]=%f,dev[0].targ[1]=%f\n",in[1],targ[1]);
					//printf("begin to run train_bunch_single\n");
					
					train_bunch_single(frames_this_bunch, realin[0], realtarg[0]);//[0]��ʾ��0��cuda device,//realin[0], realtarg[0]
			    //������ÿ��batch��ȥ�ܣ���realin��realtarg����ÿ��ָ��GPU���ÿ��batch
			    //printf("complete train_bunch_single\n");
			    }
			//else
				//train_bunch_multi(frames_this_bunch, realin, realtarg);
		}
		else
		{
			printf("this bunch has only %d samples and is ignored.\n",frames_this_bunch);
		}
		
		for(j= 0; j< GPU_selected;j++)	
		{	
			realin[j] += n_input * frames_this_bunch/GPU_selected;
			realtarg[j] += out_dims * frames_this_bunch/GPU_selected;
		}

	}
	//printf("end here before\n");
	delete[] realin;
	delete[] realtarg;
//printf("end here\n");
}

////void BP_GPU::train(int n_frames, const float* in, const int *targ)
////����д��������Ķ��GPUȥ�ܵĳ���ע�͵������ⷢ������
//void BP_GPU::train(int n_frames, const float* in, const float *targ)////////////////////////////by yongxu
//{
//
//	int i,t;
//	int frames_this_bunch;	// Number of frames to handle this bunch
//	int n_input = layersizes[0];
//	float *realin = new float[GPU_selected];
//	//int **realtarg = new int*[GPU_selected];
//	float *realtarg = new float[GPU_selected];///////////////////////////////////by yongxu
//	//float *realin;
//	//int *realtarg;
//
//	int n_frames_part = n_frames/1;
//
//// for (t=0;t<560;t++)//����check�ˣ�ƴ֡��ѵ������֡��Ӧtargetһ֡������
//// { printf("in[%d]=%f,",t,in[t]);
////   }
////   printf ("\n");
////   
////   for (t=0;t<200;t++)
////   {printf("targ[%d]=%f,",t,targ[t]);
////   	}
////   printf ("\n");
//   
//	// First copy data to GPU
//		cudaSetDevice(0);
//		todev_vf_vf("in",n_frames_part * n_input, in + 0* n_frames_part* n_input, dev[0].in, streams[0]);
//		//todev_vi_vi("targ", n_frames_part, targ + i* n_frames_part, dev[i].targ, streams[i]);
//	    todev_vf_vf("targ", n_frames_part * out_dims, targ + 0* n_frames_part * out_dims, dev[0].targ, streams[0]);
//
//	printf("Copy Data Sucess , %d Frames\n",n_frames);
//
//		realin = dev[0].in;
//		realtarg = dev[0].targ;  
//
//	printf("GPU_selected : %d\n",GPU_selected);
//	for (i=0; i< n_frames; i+= bunchsize)
//	{
//		printf("i=%d\n",i);
//		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;
//		if(frames_this_bunch == bunchsize)
//		{
//			printf("in \n");
//
//					//printf("realin[0]=%f,realtarg[0]=%f\n",realin[0],realtarg[0]);//����ط��䲻������Ҳ������
//					//printf("dev[0].in[1]=%f,dev[0].targ[1]=%f\n",in[1],targ[1]);
//					printf("begin to run train_bunch_single\n");
//					
//					train_bunch_single(frames_this_bunch, realin, realtarg);//[0]��ʾ��0��cuda device
//			    printf("complete train_bunch_single\n");
//			    
//
//		}
//		else
//		{
//			printf("this bunch has only %d samples and is ignored.\n",frames_this_bunch);
//		}
//		
//
//			realin += n_input * frames_this_bunch/1;
//			realtarg += out_dims * frames_this_bunch/1;
//		
//
//	}
//	printf("this train end\n");
//	delete[] realin;
//	delete[] realtarg;
//	
//	printf("this train end 2 \n");
//
//}

//int BP_GPU::CrossValid(int n_frames, const float* in, const int *targ)
float BP_GPU::CrossValid(int n_frames, const float* in, const float *targ)/////////////////////////////////////by yongxu
{
	//only use one GPU
	//int correct_samples =0;
	float squared_err=0.0f;/////////////////////////////////////////////by yongxu
	//float squared_err_speech=0.0f;/////////////////////////////////////////////by yongxu,�������
	//float squared_err_noise=0.0f;/////////////////////////////////////////////by yongxu���������
	//int *out = new int [bunchsize];
	int out_dims= layersizes[numlayers-1];
	
  float *out = new float [bunchsize*out_dims];///////////////////////////////by yongxu, ����ط���һ����ά������batch*feadim��
	//int *out;
	//cudaMallocHost((void**)&out, bunchsize * sizeof(int));
	int i,j,d;
	//int t;
	int frames_this_bunch;	// Number of frames to handle this bunch
	int n_input = layersizes[0];//���������ά������������չ֡�ģ�
	float *realin;


//
// for (t=0;t<560;t++)//����check�ˣ�ƴ֡��ѵ������֡��Ӧtargetһ֡������
// { printf("in[%d]=%f,",t,in[t]);
//   }
//   printf ("\n");
//   
//   for (t=0;t<200;t++)
//   {printf("targ[%d]=%f,",t,targ[t]);
//   	}
//   printf ("\n");


	// First copy data to GPU
	cudaSetDevice(0);
	//cudaSetDevice(1);
	todev_vf_vf("in", n_frames* n_input, in, dev[0].in, streams[0]);

	realin = dev[0].in;

	FILE *fp=fopen("CV_out.txt","w");

	for (i=0; i< n_frames; i+= bunchsize)//n_frames�Ǹ�CV������֡����bunchsizeָ����һ��bunch���ж���֡��Ȼ��ÿ��bunch�ֱ����
	{
		
		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;

		//cv_bunch_single(frames_this_bunch, realin, out[i]);
		cv_bunch_single(frames_this_bunch, realin, out);

		//// compute correct_samples
		////////////compute squared error
		//fprintf(fp,"%d\n\n",frames_this_bunch);
		for(j =0; j< frames_this_bunch;j++)
		{
		  for(d=350;d<out_dims;d++)///////////////////////////////////////by yongxu, ���ǵ��������ʼ����out_dimsά
		  {
		    squared_err = squared_err + (out[j*out_dims+d]-targ[j*out_dims+d])*(out[j*out_dims+d]-targ[j*out_dims+d]);/////////////by yongxu, �ر�ע�⣺squared error����correct_samples�෴�ĸ���
			
			//fprintf(fp,"%f ",out[j*out_dims+d]);
			}
			
//		  //�����������ֵ�error
//					  for(d=0;d<out_dims/2;d++)///////////////////////////////////////by yongxu, ���ǵ��������ʼ����out_dimsά
//		  {
//		    squared_err_speech = squared_err_speech + (out[j*out_dims+d]-targ[j*out_dims+d])*(out[j*out_dims+d]-targ[j*out_dims+d]);/////////////by yongxu, �ر�ע�⣺squared error����correct_samples�෴�ĸ���
//			}
//			
//					  //�����������ֵ�error
//					  for(d=out_dims/2;d<out_dims;d++)///////////////////////////////////////by yongxu, ���ǵ��������ʼ����out_dimsά
//		  {
//		    squared_err_noise = squared_err_noise + (out[j*out_dims+d]-targ[j*out_dims+d])*(out[j*out_dims+d]-targ[j*out_dims+d]);/////////////by yongxu, �ر�ע�⣺squared error����correct_samples�෴�ĸ���
//			}
			
			//fprintf(fp,"\n");
		}

		realin += n_input * frames_this_bunch;
		targ += out_dims * frames_this_bunch;
	}

	fclose(fp);

	delete []out;
	//cudaFreeHost(out);
	//return correct_samples;
	return squared_err;
}

//int BP_GPU::CrossValid(int n_frames, const float* in, const int *targ)
float BP_GPU::CrossValid2(int n_frames, const float* in, const float *targ)/////////////////////////////////////by yongxu
{
	//only use one GPU
	//int correct_samples =0;
	//float squared_err=0.0f;/////////////////////////////////////////////by yongxu
	float squared_err_speech=0.0f;/////////////////////////////////////////////by yongxu,�������
	//float squared_err_noise=0.0f;/////////////////////////////////////////////by yongxu���������
	//int *out = new int [bunchsize];
	int out_dims= layersizes[numlayers-1];
	
  float *out = new float [bunchsize*out_dims];///////////////////////////////by yongxu, ����ط���һ����ά������batch*feadim��
	//int *out;
	//cudaMallocHost((void**)&out, bunchsize * sizeof(int));
	int i,j,d;
	//int t;
	int frames_this_bunch;	// Number of frames to handle this bunch
	int n_input = layersizes[0];//���������ά������������չ֡�ģ�
	float *realin;


//
// for (t=0;t<560;t++)//����check�ˣ�ƴ֡��ѵ������֡��Ӧtargetһ֡������
// { printf("in[%d]=%f,",t,in[t]);
//   }
//   printf ("\n");
//   
//   for (t=0;t<200;t++)
//   {printf("targ[%d]=%f,",t,targ[t]);
//   	}
//   printf ("\n");


	// First copy data to GPU
	cudaSetDevice(0);
	//cudaSetDevice(1);
	todev_vf_vf("in", n_frames* n_input, in, dev[0].in, streams[0]);

	realin = dev[0].in;

	FILE *fp=fopen("CV_out.txt","w");

	for (i=0; i< n_frames; i+= bunchsize)//n_frames�Ǹ�CV������֡����bunchsizeָ����һ��bunch���ж���֡��Ȼ��ÿ��bunch�ֱ����
	{
		
		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;

		//cv_bunch_single(frames_this_bunch, realin, out[i]);
		cv_bunch_single(frames_this_bunch, realin, out);

		//// compute correct_samples
		////////////compute squared error
		//fprintf(fp,"%d\n\n",frames_this_bunch);
		for(j =0; j< frames_this_bunch;j++)
		{
//		  for(d=0;d<out_dims;d++)///////////////////////////////////////by yongxu, ���ǵ��������ʼ����out_dimsά
//		  {
//		    squared_err = squared_err + (out[j*out_dims+d]-targ[j*out_dims+d])*(out[j*out_dims+d]-targ[j*out_dims+d]);/////////////by yongxu, �ر�ע�⣺squared error����correct_samples�෴�ĸ���
//			
//			//fprintf(fp,"%f ",out[j*out_dims+d]);
//			}
			
		  //�����������ֵ�error
					  for(d=0;d<257;d++)///////////////////////////////////////by yongxu, ���ǵ��������ʼ����out_dimsά
		  {
		    squared_err_speech = squared_err_speech + (out[j*out_dims+d]-targ[j*out_dims+d])*(out[j*out_dims+d]-targ[j*out_dims+d]);/////////////by yongxu, �ر�ע�⣺squared error����correct_samples�෴�ĸ���
			}
			
//					  //�����������ֵ�error
//					  for(d=out_dims/2;d<out_dims;d++)///////////////////////////////////////by yongxu, ���ǵ��������ʼ����out_dimsά
//		  {
//		    squared_err_noise = squared_err_noise + (out[j*out_dims+d]-targ[j*out_dims+d])*(out[j*out_dims+d]-targ[j*out_dims+d]);/////////////by yongxu, �ر�ע�⣺squared error����correct_samples�෴�ĸ���
//			}
			
			//fprintf(fp,"\n");
		}

		realin += n_input * frames_this_bunch;
		targ += out_dims * frames_this_bunch;
	}

	fclose(fp);

	delete []out;
	//cudaFreeHost(out);
	//return correct_samples;
	return squared_err_speech;
}


//void BP_GPU::train_bunch_single(int n_frames, const float *in, const int* targ)
//void BP_GPU::train_bunch_single(int n_frames, const float *in, const float* targ)/////////////////////by yongxu
void BP_GPU::train_bunch_single(int n_frames, float *in, const float* targ)
{
	const float one  = 1.0f;
	const float zero = 0.0f;
	//int i,j;
	int cur_layer;			// The index of the current layer.
	int prev_layer;			// The index of the previous layer.
	int cur_layer_units;	// The number of units in the current layer.
	int prev_layer_units;	// The number of units in the previous layer.
	int cur_layer_size;		// The size of the current layer.
  int prev_layer_size;
  
	float* cur_layer_x;
	float* cur_layer_y;				// Output from the current layer
	const float* prev_layer_y;	// Output from the previous non-linearity.
	float* cur_layer_dydx;	// dydx for the current layer.
	float* cur_layer_dedy;	// dedy for the current layer.
	float* prev_layer_dedy;	// dedy for the previous layer.
	float* cur_layer_dedx;	// dedx for the current layer.
	float* cur_layer_ydedx;
	float* cur_layer_sumdedx;
	float* cur_layer_bias;	// Biases for the current layer.
	float* cur_layer_delta_bias; // Delta biases for the current layer.
	float* cur_layer_delta_weights;
	float* cur_weights;		// Weights inputing to the current layer.
	float cur_lrate =  lrate;
	
	//float *out_check = new float [n_frames*out_dims];//Ϊ��check��������

 // printf("in train_bunch_single\n");
  //FILE *fp=fopen("log_train_bunch_single.txt","w");//���������д���������ѵ�����Ϊ��cuda�����Ҫ����cpu����У�


	//// Forward
	for (cur_layer=1; cur_layer< numlayers; cur_layer++)
	{
		//printf("forward ing\n");
		prev_layer = cur_layer - 1;
		cur_layer_units = layersizes[cur_layer];
		prev_layer_units = layersizes[prev_layer];
		cur_layer_size = cur_layer_units * n_frames;//batch���֡��
		prev_layer_size = prev_layer_units * n_frames;
		cur_layer_x = dev[0].layer_x[cur_layer];
		cur_layer_y = dev[0].layer_y[cur_layer];
		
		//if (cur_layer==1)//Ϊ�����dropout����ע�͵���
		//	prev_layer_y = in;
		//else
		//	prev_layer_y = dev[0].layer_y[prev_layer];
		
				if (cur_layer==1)
		{   
			if(dropoutflag==1)
			{
			  curandGenerateUniform(gen[0], dev[0].DevRandVector, prev_layer_size);
			  DevDropout(streams[0],prev_layer_size,visible_omit,in,dev[0].DevRandVector);
			}
			prev_layer_y = in;
		}
		else
		{
			if(dropoutflag==1)
			{
			 curandGenerateUniform(gen[0], dev[0].DevRandVector, prev_layer_size);
			 DevDropout(streams[0],prev_layer_size, hid_omit, dev[0].layer_y[prev_layer], dev[0].DevRandVector);
			}
			prev_layer_y = dev[0].layer_y[prev_layer];
        }
	    cudaDeviceSynchronize();
		
		cur_layer_bias = dev[0].bias[cur_layer];
		cur_weights = dev[0].weights[cur_layer];

		DevMultiCopy(streams[0],n_frames, cur_layer_units, cur_layer_bias, cur_layer_x);
		SgemmNN(handles[0],cur_layer_units, prev_layer_units, n_frames, cur_weights, prev_layer_y, cur_layer_x, one, one); 

		if (cur_layer != numlayers - 1){
			DevSigmoid(streams[0],cur_layer_size, cur_layer_x, cur_layer_y);
		}
		else{  /////////////////////////////ֱ��ע�͵�������ؾ���linear�ģ�����
			//DevSoftmax(streams[0],n_frames, cur_layer_units, cur_layer_x, dev[0].out);
			//DevLinearOutCopy(streams[0],n_frames, cur_layer_units, cur_layer_x, dev[0].out);
	    //out=cur_layer_x;
	    //cudaSetDevice(0);
	    //printf("come here\n");
	    cudaMemcpy(dev[0].out,cur_layer_x,n_frames*cur_layer_units*sizeof(float),cudaMemcpyDeviceToDevice);
	    //cudaMemcpy(out_check,cur_layer_x,n_frames*cur_layer_units*sizeof(float),cudaMemcpyDeviceToHost);
		
		//����������
//		for(i =0; i< n_frames;i++)
//		{
//		  for(j=0;j<out_dims;j++)///////////////////////////////////////by yongxu, ���ǵ��������ʼ����out_dimsά
//		  {
//			
//			printf("%f ",out_check[i*out_dims+j]);
//			}
//			printf("\n");exit(0);
//		}
		  //delete []out;
		}
	}

	// Backward
	for (cur_layer = numlayers -1; cur_layer >0; cur_layer--)
	{
		//printf("Backward ing\n");
		prev_layer = cur_layer - 1;
		cur_layer_units = layersizes[cur_layer];
		prev_layer_units = layersizes[prev_layer];
		cur_layer_size = cur_layer_units * n_frames;
		cur_layer_y = dev[0].layer_y[cur_layer];
		if (cur_layer==1)
			prev_layer_y = in;
		else
			prev_layer_y = dev[0].layer_y[prev_layer];
		cur_layer_dydx = dev[0].layer_dydx[cur_layer];
		cur_layer_dedy = dev[0].layer_dedy[cur_layer];
		prev_layer_dedy = dev[0].layer_dedy[prev_layer];
		cur_layer_dedx = dev[0].layer_dedx[cur_layer];
		cur_layer_ydedx = dev[0].layer_ydedx[cur_layer];
		cur_layer_sumdedx = dev[0].layer_sumdedx[cur_layer];
		cur_layer_bias = dev[0].bias[cur_layer];
		cur_layer_delta_bias = dev[0].delta_bias[cur_layer];
		cur_layer_delta_weights = dev[0].delta_weights[cur_layer];
		cur_weights = dev[0].weights[cur_layer];

		if (cur_layer != numlayers - 1)
		{
			//printf("former layers' sigmoid\n");
			DevDsigmoid(streams[0], cur_layer_size, cur_layer_y, cur_layer_dydx);
			DevVecMul(streams[0],   cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
		}
		//else
		//{
		
		//DevSubIndex(streams[0], n_frames, cur_layer_units, dev[0].out, targ, cur_layer_dedx);
		//��cpu���Ƶ�gpu
	  //    DevLinearOutCopy(streams[0], n_frames, cur_layer_units, dev[0].out, targ, cur_layer_dedx);
		//}
		//��ƽ������󵼣�//////////////////////////////////////////yongxu
		else
		{
		  //printf("begin to cal squared error\n");
		  //printf("targ[0]=%f,targ[1]=%f\n",targ[0],targ[1]);
			//DevSubClean(streams[0], n_frames, cur_layer_units, dev[0].layer_x[numlayers - 1], targ, cur_layer_dedx);
		  DevSubClean(streams[0], n_frames, cur_layer_units, dev[0].out, targ, cur_layer_dedx);
		  //dev[0].layer_x[numlayers - 1]
		}

		if (cur_layer != 1)
		{
			SgemmTN(handles[0], prev_layer_units, cur_layer_units, n_frames, cur_weights, cur_layer_dedx, prev_layer_dedy, zero, one);
		}

		// Update weights.
		//printf("Update weights\n");
		//SgemmNT(handles[0], cur_layer_units, n_frames, prev_layer_units, cur_layer_dedx, prev_layer_y, cur_layer_delta_weights ,momentum, -cur_lrate/n_frames);
		SgemmNT(handles[0], cur_layer_units, n_frames, prev_layer_units, cur_layer_dedx, prev_layer_y, cur_layer_ydedx ,zero, one);
		updatedelta(streams[0], cur_layer_units * prev_layer_units, cur_layer_delta_weights, cur_weights, cur_layer_ydedx, n_frames, momentum, cur_lrate, weightcost);
		//cublasSaxpy(handles[0],cur_layer_units *prev_layer_units, &cur_lr_wc, cur_weights,1,cur_layer_delta_weights ,1);

		//DevAccSumrow(streams[0], cur_layer_units, n_frames, cur_layer_dedx, cur_layer_delta_bias, momentum, -cur_lrate/n_frames);
		DevAccSumrow(streams[0], cur_layer_units, n_frames, cur_layer_dedx, cur_layer_sumdedx, zero, one);
		updatedelta(streams[0], cur_layer_units, cur_layer_delta_bias, cur_layer_bias, cur_layer_sumdedx, n_frames, momentum, cur_lrate, zero);
		//cublasSaxpy(handles[0],cur_layer_units, &cur_lr_wc, cur_layer_bias,1,cur_layer_delta_bias ,1);

		DevAccSum(streams[0],	cur_layer_units *prev_layer_units, cur_layer_delta_weights,	cur_weights, 1.0);		
		DevAccSum(streams[0],	cur_layer_units, cur_layer_delta_bias,	cur_layer_bias, 1.0);

		///
		/*
		if(cur_layer ==1){
		float *tmpout = new float[1 *cur_layer_units];
		fromdev_vf_vf("data",1 *cur_layer_units, cur_layer_bias,tmpout, streams[0]);
		for(int tmpj =0 ;tmpj < cur_layer_units ;tmpj ++)
		{
		for(int tmpi =0;tmpi< 1; tmpi++)
		{
		printf("%f\n",(tmpout[tmpj + tmpi *cur_layer_units]));
		}
		}
		delete [] tmpout;
		exit(0);}
		*/
		///
  //printf("come to end\n");
	}
	//fclose(fp);
}

//void BP_GPU::cv_bunch_single(int n_frames, const float *in, int* out)
void BP_GPU::cv_bunch_single(int n_frames, const float *in, float* out)///////////////////////////////by yongxu
{
    

	const float one  = 1.0f;
	//const float zero = 0.0f;
	//int i,j;
	int cur_layer;			// The index of the current layer.
	int prev_layer;			// The index of the previous layer.
	int cur_layer_units;	// The number of units in the current layer.
	int prev_layer_units;	// The number of units in the previous layer.
	int cur_layer_size;		// The size of the current layer.
  int out_dims= layersizes[numlayers-1];
  
	float* cur_layer_x;
	float* cur_layer_y;				// Output from the current layer
	const float* prev_layer_y;	// Output from the previous non-linearity.
	float* cur_layer_bias;	// Biases for the current layer.
	float* cur_weights;		// Weights inputing to the current layer.

	//int *devout;
	//devnew_vi("devout", n_frames, &devout);
	float *devout;/////////////////////////////////by yongxu
	devnew_vf("devout", n_frames*out_dims, &devout);
	
	//dropout����
	int weight_size;
	    float vis_keep;
	float hid_keep;
    vis_keep=1.0f-visible_omit;
	hid_keep=1.0f-hid_omit;
	
	//// Forward
	for (cur_layer=1; cur_layer< numlayers; cur_layer++)
	{
		prev_layer = cur_layer - 1;
		cur_layer_units = layersizes[cur_layer];
		prev_layer_units = layersizes[prev_layer];
		cur_layer_size = cur_layer_units * n_frames;
		cur_layer_x = dev[0].layer_x[cur_layer];
		cur_layer_y = dev[0].layer_y[cur_layer];
		
		 weight_size=prev_layer_units*cur_layer_units;
		
		if (cur_layer==1)
			prev_layer_y = in;
		else
			prev_layer_y = dev[0].layer_y[prev_layer];
		cur_layer_bias = dev[0].bias[cur_layer];
		
				if (dropoutflag==1)
		{
			if(cur_layer==1)
			   DevWeightMultiP(streams[0], weight_size, vis_keep, dev[0].weights[cur_layer]);
			else
				DevWeightMultiP(streams[0], weight_size, hid_keep, dev[0].weights[cur_layer]);
		}
		
		cur_weights 	 = dev[0].weights[cur_layer];

		DevMultiCopy(streams[0],n_frames, cur_layer_units, cur_layer_bias, cur_layer_x);
		SgemmNN(handles[0],cur_layer_units, prev_layer_units, n_frames, cur_weights, prev_layer_y, cur_layer_x, one, one); 
    
    		if (dropoutflag==1)
		{
            if(cur_layer==1)
			   DevWeightMultiP(streams[0], weight_size, 1.0f/vis_keep, dev[0].weights[cur_layer]);
			else 
				DevWeightMultiP(streams[0], weight_size, 1.0f/hid_keep, dev[0].weights[cur_layer]);

		}
    
		if (cur_layer != numlayers - 1){
			DevSigmoid(streams[0],cur_layer_size, cur_layer_x, cur_layer_y);
		}
		else{  /////////////////////////////////////////yongxu ע�͵��Ϳ��Եõ�һ���������������
		//	DevSoftmax(streams[0],n_frames, cur_layer_units, cur_layer_x, dev[0].out);
		//	DevGetMaxIndex(streams[0], cur_layer_units, n_frames, dev[0].out,  devout);
		//DevLinearOutCopy(streams[0],n_frames, cur_layer_units, cur_layer_x, dev[0].out);
		//cudaMemcpy(dev[0].out,cur_layer_x,n_frames*cur_layer_units*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaMemcpy(devout,cur_layer_x,n_frames*cur_layer_units*sizeof(float),cudaMemcpyDeviceToDevice);
		}
	}
	//fromdev_vi_vi("devout",n_frames,devout,out, streams[0]);
	//devfree_vi("devout",devout);/////////////////////////////////////////yongxu
    fromdev_vf_vf("devout",n_frames*out_dims,devout,out, streams[0]);
	devfree_vf("devout",devout);

	////
	//		float *asf = new float[cur_layer_units* n_frames];
	//		//fromdev_vf_vf("out", cur_layer_units* n_frames, dev[0].out ,asf, streams[0]);
	//		for(int tmp=0;tmp <  n_frames;tmp++)
	//   		printf("%d\n",out[tmp]);
	//  		delete []asf;
	//   		exit(0);

}

////void BP_GPU::train_bunch_multi(int n_frames, float **in, int** targ)
//void BP_GPU::train_bunch_multi(int n_frames, float **in, float** targ)/////////////////////yongxu
//{
//	const float one  = 1.0f;
//	const float zero = 0.0f;
//	int i;
//	int cur_layer;			// The index of the current layer.
//	int prev_layer;			// The index of the previous layer.
//	
//	float cur_lrate =  lrate;
//
//	int n_frames_part[GPU_selected];
//	int part = bunchsize/GPU_selected;
//
//	for(i= 0; i< GPU_selected;i++)
//	{
//		n_frames_part[i] = part;
//	}
//	n_frames_part[GPU_selected -1] = n_frames -part*(GPU_selected -1);
//
//	for(i=0;i<GPU_selected;i++)
//	{
//		cudaSetDevice(i);
//		//// Forward
//		for (cur_layer=1; cur_layer< numlayers; cur_layer++)
//		{
//	
//			prev_layer = cur_layer - 1;
//			DevMultiCopy(streams[i], n_frames_part[i], layersizes[cur_layer], dev[i].bias[cur_layer], dev[i].layer_x[cur_layer]);
//			if (cur_layer==1)       
//                SgemmNN(handles[i], layersizes[cur_layer], layersizes[prev_layer], n_frames_part[i], dev[i].weights[cur_layer], in[i], dev[i].layer_x[cur_layer], one, one); 
//			else
//			    SgemmNN(handles[i], layersizes[cur_layer], layersizes[prev_layer], n_frames_part[i], dev[i].weights[cur_layer], dev[i].layer_y[prev_layer], dev[i].layer_x[cur_layer], one, one); 
//			
//			if (cur_layer != numlayers - 1){
//				DevSigmoid(streams[i], layersizes[cur_layer] * n_frames_part[i], dev[i].layer_x[cur_layer], dev[i].layer_y[cur_layer]);
//			}
//			//else{ /////////////////////////////////yongxu, ע�͵����ܵõ����������
//			//	DevSoftmax(streams[i],n_frames_part[i], layersizes[cur_layer], dev[i].layer_x[cur_layer], dev[i].out);
//			//}
//			
//
//		}
//
//		// Backward
//		for (cur_layer = numlayers -1; cur_layer >0; cur_layer--)
//		{
//			prev_layer = cur_layer - 1;
//
//
//			if (cur_layer != numlayers - 1)
//			{
//				DevDsigmoid(streams[i], layersizes[cur_layer] * n_frames_part[i], dev[i].layer_y[cur_layer], dev[i].layer_dydx[cur_layer]);
//				DevVecMul(streams[i],   layersizes[cur_layer] * n_frames_part[i], dev[i].layer_dydx[cur_layer], dev[i].layer_dedy[cur_layer], dev[i].layer_dedx[cur_layer]);
//
//			}
//			//else/////////////////////////////////yongxu, ע�͵����ܵõ����������
//			//{
//			//
//			//	DevSubIndex(streams[i], n_frames_part[i], layersizes[cur_layer], dev[i].out, targ[i], dev[i].layer_dedx[cur_layer]);
//			//	
//			//}
//		    //��ƽ������󵼣�//////////////////////////////////////////yongxu
//		    else
//		    {
//		
//		    	DevSubClean(streams[i], n_frames_part[i], layersizes[cur_layer], dev[i].layer_x[numlayers - 1], targ[i], dev[i].layer_dedx[cur_layer]);
//		
//		     }
//
//			if (cur_layer != 1)
//			{
//				SgemmTN(handles[i], layersizes[prev_layer], layersizes[cur_layer], n_frames_part[i], dev[i].weights[cur_layer], dev[i].layer_dedx[cur_layer], dev[i].layer_dedy[prev_layer], zero, one);
//				
//			}
//
//			// Update weights.
//			if (cur_layer ==1)
//		        SgemmNT(handles[i], layersizes[cur_layer], n_frames_part[i], layersizes[prev_layer], dev[i].layer_dedx[cur_layer], in[i], dev[i].layer_ydedx[cur_layer] ,zero, one);
//			else
//			    SgemmNT(handles[i], layersizes[cur_layer], n_frames_part[i], layersizes[prev_layer], dev[i].layer_dedx[cur_layer], dev[i].layer_y[prev_layer], dev[i].layer_ydedx[cur_layer] ,zero, one);
//		    DevAccSumrow(streams[i], layersizes[cur_layer], n_frames_part[i], dev[i].layer_dedx[cur_layer], dev[i].layer_sumdedx[cur_layer], zero, one);
//
//		}
//	}
//	cudaDeviceSynchronize();
//	cudaSetDevice(0);
//	
//	for(i= 1; i< GPU_selected;i++)
//	{
//		cudaDeviceEnablePeerAccess(i, 0);
//		for (cur_layer=1; cur_layer< numlayers; cur_layer++)
//		{
//		    prev_layer = cur_layer - 1;
//
//			cublasSaxpy(handles[0],layersizes[cur_layer] * layersizes[prev_layer], &one, dev[i].layer_ydedx[cur_layer], 1, dev[0].layer_ydedx[cur_layer] , 1);
//			cublasSaxpy(handles[0],layersizes[cur_layer], &one, dev[i].layer_sumdedx[cur_layer], 1, dev[0].layer_sumdedx[cur_layer] , 1);
//
//		} 
//	}
//	cudaDeviceSynchronize();
//	for (cur_layer=1; cur_layer< numlayers; cur_layer++)
//	{
//		prev_layer = cur_layer - 1;
//
//		updatedelta(streams[0], layersizes[cur_layer] * layersizes[prev_layer], dev[0].delta_weights[cur_layer], dev[0].weights[cur_layer], dev[0].layer_ydedx[cur_layer], n_frames, momentum, cur_lrate, weightcost);
//		updatedelta(streams[0], layersizes[cur_layer], dev[0].delta_bias[cur_layer], dev[0].bias[cur_layer], dev[0].layer_sumdedx[cur_layer], n_frames, momentum, cur_lrate, zero);
//		DevAccSum(streams[0],	layersizes[cur_layer] * layersizes[prev_layer], dev[0].delta_weights[cur_layer],	dev[0].weights[cur_layer], 1.0);		
//		DevAccSum(streams[0],	layersizes[cur_layer], dev[0].delta_bias[cur_layer],	dev[0].bias[cur_layer], 1.0);
//	}
//	//cudaStreamSynchronize(streams[0]);
//
//	////copy paras to other gpus
//	for(i= 1; i< GPU_selected;i++)
//	{
//	    //cudaSetDevice(i);
//		//cudaDeviceEnablePeerAccess(i, 0);
//		for (cur_layer=1; cur_layer< numlayers; cur_layer++)
//		{
//			prev_layer = cur_layer - 1;
//
//			cublasScopy(handles[0], layersizes[cur_layer] * layersizes[prev_layer], dev[0].weights[cur_layer],1,dev[i].weights[cur_layer] ,1);
//			cublasScopy(handles[0], layersizes[cur_layer], dev[0].bias[cur_layer],1, dev[i].bias[cur_layer] ,1);
//
//
//			cublasScopy(handles[0],layersizes[cur_layer] * layersizes[prev_layer], dev[0].delta_weights[cur_layer],1,dev[i].delta_weights[cur_layer] ,1);
//			cublasScopy(handles[0],layersizes[cur_layer], dev[0].delta_bias[cur_layer],1, dev[i].delta_bias[cur_layer] ,1);
//		}			
//
//	}
//	cudaStreamSynchronize(streams[0]);
//	cudaDeviceSynchronize();
//
//}

void BP_GPU::returnWeights(float **weights, float **bias)
{
	int i;
	////copy weights && biases to devices

	cudaSetDevice(0);
  //cudaSetDevice(1);
   
	for(i = 1; i< numlayers; i++)
	{
		fromdev_vf_vf("weights", layersizes[i-1] *layersizes[i], dev[0].weights[i], weights[i], streams[0]);
		fromdev_vf_vf("bias", layersizes[i], dev[0].bias[i], bias[i], streams[0]);
	}
}

///// following are alloc and free functions
void BP_GPU::devnew_vf(const char* varname, int n, float **devptr)
{
	cudaError_t cudaStat =  cudaMalloc((void **) devptr, n* sizeof(float));
	if(cudaStat !=cudaSuccess ) 
	{
		printf("%s device momory alloc error\n", varname);
		exit(0);
	}
	//float *zero = new float [n];
	float *zero;
	cudaMallocHost((void**)&zero,n*sizeof(float));

	for(int i=0;i< n;i++)
		zero[i] = 0.0f;
	cublasSetVector(n,sizeof(float),zero,1,(*devptr),1);
	//delete []zero;
	cudaFreeHost(zero);
}

void BP_GPU::devnew_vi(const char* varname, int n, int **devptr)
{
	cudaError_t cudaStat = cudaMalloc((void **) devptr, n* sizeof(int));
	if(cudaStat !=cudaSuccess ) 
	{
		printf( "%s device momory alloc error\n", varname);
		exit(0);
	}
	//int *zero = new int [n];
	int *zero;
	cudaMallocHost((void**)&zero,n*sizeof(int));

	for(int i=0;i< n;i++)
		zero[i] = 0;
	cublasSetVector(n,sizeof(int),zero,1,(*devptr),1);
	//delete []zero;
	cudaFreeHost(zero);
}

void BP_GPU::devfree_vf(const char* varname, float* devptr)
{
	cudaFree((void *) devptr);
}

void BP_GPU::devfree_vi(const char* varname, int* devptr)
{
	cudaFree((void *) devptr);
}

void BP_GPU::todev_vf_vf(const char* varname, int n, const float* from, float* devto, cudaStream_t stream)
{
	cublasStatus_t  e = cublasSetVectorAsync(n, sizeof(float), from, 1, devto, 1, stream);
	if (e != CUBLAS_STATUS_SUCCESS)
	{
		printf("cuda blas todev_vf_vf error variable %s\n",varname);
		exit(0);
	}
}

void BP_GPU::fromdev_vf_vf(const char* varname, int n, const float* devfrom, float* to, cudaStream_t stream)
{
	cublasStatus_t e = cublasGetVectorAsync(n, sizeof(float), devfrom, 1, to, 1, stream);
	if (e != CUBLAS_STATUS_SUCCESS)
	{
		printf("cuda blas fromdev_vf_vf error variable %s\n",varname);
		exit(0);
	}
}

//void BP_GPU::todev_vi_vi(const char* varname, int n, const int* from,int *devto, cudaStream_t stream)
//{
//	cublasStatus_t e = cublasSetVectorAsync(n, sizeof(int), from, 1, devto, 1, stream);
//	if (e != CUBLAS_STATUS_SUCCESS)
//	{
//		printf("cuda blas todev_vi_vi error variable %s\n", varname);
//		exit(0);
//	}
//}

//void BP_GPU::fromdev_vi_vi(const char* varname, int n,const int* devfrom, int* to, cudaStream_t stream)
//{
//	cublasStatus_t e = cublasGetVectorAsync(n, sizeof(int), devfrom, 1, to, 1, stream);
//	if (e != CUBLAS_STATUS_SUCCESS)
//	{
//		printf("cuda blas fromdev_vi_vi error variable %s\n", varname);
//		exit(0);
//	}
//}
