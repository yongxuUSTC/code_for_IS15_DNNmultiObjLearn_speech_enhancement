///////////////////////////
////  yongxu        ///////        
////  iflytek lab      ///////
////  2013/8/28//////
///////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <time.h>


#include "BP_GPU.h"
#include "Interface.h"

void test();
void test2(int argc,char*argv[]);

int main(int argc, char *argv[])
{
	
	struct WorkPara *paras;
	int *chunk_index;//һ���Զ����ڴ�Ĵ�pfile���֡����chunk�� index
	int cur_chunk_samples;
	int i,j;
	//int correct_samples = 0;
	float squared_err=0.0;///////////////////////////////yong xu
	float squared_err_speech=0.0;///////////////////////////////yong xu
	double timenow;
	timenow = time(NULL);
	
	Interface *InterObj = new Interface;
	InterObj->Initial(argc, argv);
	paras = InterObj->para;
	BP_GPU *TrainObj = new BP_GPU(paras->gpu_used, InterObj->numlayers, paras->layersizes, paras->bunchsize,  paras->lrate, paras->momentum,
				paras->weightcost, paras->weights, paras->bias,paras->dropoutflag,paras->visible_omit,paras->hid_omit);

  
	InterObj->get_pfile_info();
	//fprintf(InterObj->fp_log,"complete to read pfile_info\n");
	
	/////////train	
	InterObj->get_chunk_info(paras->train_sent_range);
	//fprintf(InterObj->fp_log,"complete to read chunk_info\n");
	
	chunk_index = new int [InterObj->total_chunks];
	for(i=0;i< InterObj->total_chunks;i++){
		chunk_index[i] =i;
	}
	
	InterObj->GetRandIndex(chunk_index,InterObj->total_chunks);
	for(i=0;i< InterObj->total_chunks; i++){
		//printf("begin to readchunk %d......,InterObj->total_chunks=%d\n",i,InterObj->total_chunks);
		cur_chunk_samples = InterObj->Readchunk(chunk_index[i]);
		fprintf(InterObj->fp_log,"Starting chunk %d of %d containing %d samples.\n", i+1 ,InterObj->total_chunks, cur_chunk_samples);
		fflush(InterObj->fp_log);
		TrainObj->train(cur_chunk_samples, paras->indata ,paras->targ);
	}
	
	printf("begin to write weights\n");
	TrainObj->returnWeights(paras->weights,paras->bias);
	InterObj->Writeweights();
	printf("finish to write weights\n\n");
	
	printf("begin to CV\n");
	////CV
	fprintf(InterObj->fp_log,"Starting CV.\n");
	InterObj->get_chunk_info_cv(paras->cv_sent_range);
	
	//FILE *fp=fopen("CV_in.txt","w");
   //printf("cv 1 here\n");
	chunk_index = new int [InterObj->cv_total_chunks];
	for(i=0;i< InterObj->cv_total_chunks;i++){
		chunk_index[i] =i;
	}
	//printf("cv 2 here,cv_total_chunks=%d\n",InterObj->cv_total_chunks);
	for(i=0;i< InterObj->cv_total_chunks; i++){
		cur_chunk_samples = InterObj->Readchunk_cv(chunk_index[i]);
		printf("cur_chunk_samples=%d\n",cur_chunk_samples);
		//correct_samples += TrainObj->CrossValid(cur_chunk_samples, paras->indata ,paras->targ);
		squared_err += TrainObj->CrossValid(cur_chunk_samples, paras->indata ,paras->targ);
		//for(j=0;j<cur_chunk_samples;j++)
		  //fprintf(fp,"%d ",paras->targ[j]);
	}
	//fclose(fp);
	//float cvacc = 100.0 *((float) correct_samples/InterObj->cv_total_samples);
	//printf("cal to cvacc\n");
	float cvacc = ((float) squared_err/InterObj->cv_total_samples);////////////////////////////////////////yongxu
	//fprintf(InterObj->fp_log,"CV over. right num: %d, ACC: %.2f%%\n", correct_samples, cvacc);
	fprintf(InterObj->fp_log,"CV over. IBM squared error: %f\n", cvacc);
	fflush(InterObj->fp_log);
	
	printf("begin to CV2 for speech part error\n");
	////CV
	fprintf(InterObj->fp_log,"Starting CV2.\n");
	//InterObj->get_chunk_info_cv(paras->cv_sent_range);
	
	//FILE *fp=fopen("CV_in.txt","w");
   //printf("cv 1 here\n");
	chunk_index = new int [InterObj->cv_total_chunks];
	for(i=0;i< InterObj->cv_total_chunks;i++){
		chunk_index[i] =i;
	}
	//printf("cv 2 here,cv_total_chunks=%d\n",InterObj->cv_total_chunks);
	for(i=0;i< InterObj->cv_total_chunks; i++){
		cur_chunk_samples = InterObj->Readchunk_cv(chunk_index[i]);
		printf("cur_chunk_samples=%d\n",cur_chunk_samples);
		//correct_samples += TrainObj->CrossValid(cur_chunk_samples, paras->indata ,paras->targ);
		squared_err_speech += TrainObj->CrossValid2(cur_chunk_samples, paras->indata ,paras->targ);
		//for(j=0;j<cur_chunk_samples;j++)
		  //fprintf(fp,"%d ",paras->targ[j]);
	}
	//fclose(fp);
	//float cvacc = 100.0 *((float) correct_samples/InterObj->cv_total_samples);
	//printf("cal to cvacc\n");
	float cvacc2 = ((float) squared_err_speech/InterObj->cv_total_samples);////////////////////////////////////////yongxu
	//fprintf(InterObj->fp_log,"CV over. right num: %d, ACC: %.2f%%\n", correct_samples, cvacc);
	fprintf(InterObj->fp_log,"CV2 over. SPEECH squared error: %f\n", cvacc2);
	fflush(InterObj->fp_log);

	delete [] chunk_index;

	timenow = time(NULL) - timenow;
	fprintf(InterObj->fp_log,"Total cost time: %.1f s.\n", timenow);
	
	printf("all finish!\n");
	
	delete TrainObj;
	delete InterObj;
	
		
	return 1;
}
/*
void test()  ////test the part of gpu training 
{
///for debug
	int i;
	FILE *fp_init_weight;
	
	float *in;
	int *targ;
	int gpu_used = 1;
	int numlayers =3;
	int layersizes[3] = {429,1024,183};
	int bunchsize	= 128;
	float momentum = 0.9;
	float lrate = 0.002;
	float weightcost = 0;
	float *weights[3];
	float *bias[3];
	char init_weightFN[] = "/home/sfxue/TIMIT/c_code/finetune/BP_GPU/test/mlp.0.wts";
	
	//// Init weights
	for(i =1; i< numlayers; i++)
	{
		int size	= layersizes[i] *layersizes[i-1];
		weights[i] = new float [size];
		bias[i] = new float [layersizes[i]];
	  
		memset(weights[i],0,size *sizeof(float));
		memset(bias[i],0,layersizes[i] *sizeof(float));
	}
	
	if(NULL ==(fp_init_weight = fopen(init_weightFN, "rb")))
	{
		printf("can not open initial weights file: %s\n", init_weightFN);
		exit(0);
	}
	else
	{
		int stat[10];
		char head[256];
	
		for(i =1; i< numlayers; i++)
		{
			fread(stat,sizeof(int),5,fp_init_weight);
			fread(head,sizeof(char),stat[4],fp_init_weight);
			
			if(stat[1] != layersizes[i] || stat[2] != layersizes[i -1])
			{
				printf("init weights node nums do not match\n");
				exit(0);
			}
			fread(weights[i],sizeof(float),layersizes[i -1] *layersizes[i],fp_init_weight);
			
			fread(stat,sizeof(int),5,fp_init_weight);
			fread(head,sizeof(char),stat[4],fp_init_weight);
			
			if(stat[2] != layersizes[i] || stat[1] != 1)
			{
				printf("init bias node nums do not match\n");
				exit(0);
			}
			fread(bias[i],sizeof(float),layersizes[i],fp_init_weight);
		}
		fclose(fp_init_weight);
	}

	BP_GPU *TrainObj = new BP_GPU(gpu_used, numlayers, layersizes, bunchsize, lrate, momentum, 
				weightcost, weights, bias);
				
		///for debug
    float *tmpin = new float[128*429];
    char *tmpname = "/home/sfxue/TIMIT/c_code/finetune/BP_GPU/test/testin.txt";
    FILE *fp_tmp = fopen(tmpname,"rt");
    
    for(int tmpi =0;tmpi< 128*429; tmpi++)
    {
    	fscanf(fp_tmp,"%f\n",&(tmpin[tmpi]));
    }
   	fclose(fp_tmp);
       
    int *tmptarg = new int[128];
    tmpname = "/home/sfxue/TIMIT/c_code/finetune/BP_GPU/test/testtarg.txt";
    fp_tmp = fopen(tmpname,"rt");
  
    for(int tmpi =0;tmpi< 128; tmpi++)
    {
    	fscanf(fp_tmp,"%d\n",&(tmptarg[tmpi]));
    }
    fclose(fp_tmp);
    
	TrainObj->train(128, tmpin ,tmptarg); 
	
	delete [] tmpin;
  delete [] tmptarg;
	delete TrainObj;
}

void test2(int argc,char*argv[]) ////test the part of Reading data 
{
///for debug
//	gpu_used=1 numlayers=5 layersizes=473,1024,1024,1024,3969 bunchsize=512 momentum=0.0002 weightcost=0.0001 lrate=0.002 initwts_file=/home/jiapan/new_BP_Code/QN_cmp/test_mlp/mlp.0.wts norm_file=/home/jiapan/Tandem_train/80H_Chinese/lib/fea_tr.norm fea_file=/home/jiapan/Tandem_train/80H_Chinese/lib/fea_tr.pfile targ_file=/home/jiapan/Tandem_train/80H_Chinese/lib/lab_state.pfile outwts_file=/home/jiapan/new_BP_Code/QN_cmp/test_mlp/mlp.test.wts log_file=/home/jiapan/new_BP_Code/QN_cmp/test_mlp/mlp.test.log train_sent_range=1-100 cv_sent_range=101-102 fea_dim=43 fea_context=11 traincache=200 init_randem_seed=6346 targ_offset=5

	Interface *testObj = new Interface;
	testObj->Initial(argc, argv);
	testObj->get_pfile_info();
	testObj->get_chunk_info(testObj->para->train_sent_range);
	testObj->Readchunk(2);
	
	delete testObj;
}
*/
/*
���ݶ�ȡ����
1. ��ȡ pfile���ܾ���������ȡ��֡�� �Լ�ÿ�仰������֡��
2. ������ʼ�䡢�����䡢ÿ�仰������֡�������������Сȷ��ÿ�� chunk����ֹ֡id �Լ��ܵ�chunk��Ŀ
3. ��chunk����������ÿ�������ȡһ��chunk�������е��ܵ�������������index,
	ͬʱ����ȡ�����ݽ���MVN������index�ʹ������һ��������е���������
*/