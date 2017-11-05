//#include "/usr/local/cuda/include/cuda_runtime.h"
//#include <cuda_runtime.h>
//#include "/usr/local/cuda/include/cublas_v2.h"
//#include <cublas_v2.h>
#include "/usr/local/cuda-5.0/include/cuda_runtime.h"
#include "/usr/local/cuda-5.0/include/cublas_v2.h"
#include "/usr/local/cuda-5.0/include/curand.h" //���������ʾcuda_runtime.h�Ҳ�������Ҫ����curand.h����cuda_runtime.h��ȫ·������ȥ
//#include <curand.h>
//#include "/usr/local/cuda/include/cuda_runtime.h"
//#include "/usr/local/cuda/include/cublas_v2.h"
//#include "/usr/local/cuda/include/curand.h"

#define MAXLAYER	10
#define MAXCACHEFRAME 200000                                

struct BP_WorkSpace
{
	float *in; 								/// input data
	float *out;								/// Output data�������float�ͣ���ͨ��softmax���0,1
	//int *targ;							  /// target data����Ҫ��#############################��target��int���øĳ�float
    float *targ;////////////////////////////////////by yongxu

	float *weights[MAXLAYER];   	/// weights for layers��ָ�����飬ÿ�����鶼��ָ�룬���������飬Ȩ���Ƕ�ά�ģ���int (*p)[10]; p��Ϊָ������Ԫ�ص�ַ��ָ�룬������ָ��
	float *bias[MAXLAYER];      	/// biases for layers��rbm�����룬�������һ��bias

	float *layer_x[MAXLAYER];  	  /// Input to layer
	float *layer_y[MAXLAYER]; 		/// Output from layer
	float *layer_dedy[MAXLAYER];  /// de/dy
	float *layer_dydx[MAXLAYER];  /// dy/dx
	float *layer_dedx[MAXLAYER];  /// de/dx
	float *layer_ydedx[MAXLAYER];
	float *layer_sumdedx[MAXLAYER]; 

	float *delta_bias[MAXLAYER]; // Output bias update
	float *delta_weights[MAXLAYER]; // Output bias update
		float *DevRandVector; //Dropout������洢����
	int *DevSeed;//Dropout���������
};

class BP_GPU
{
public:
	BP_GPU(int a_GPU_selected, int a_numlayers, int *a_layersizes, int a_bunchsize, float a_lrate, float a_momentum, float  a_weightcost,
		float **weights, float **bias,int dropoutflag, float visible_omit,float hid_omit);
	~BP_GPU();
public:
	//void train(int n_frames, const float* in, const int *targ);
	//void train(int n_frames, const float* in, const float *targ);////////////////////////////////////////by yongxu
	void train(int n_frames, float* in, const float *targ);
	//void train_bunch_multi(int n_frames,  float** in, int **targ);
	//void train_bunch_multi(int n_frames,  float** in, float **targ);//////////////////////////////////////by yongxu
	void train_bunch_multi(int n_frames,  float** in, float **targ);
	//void train_bunch_single(int n_frames, const float* in, const int *targ);
	//void train_bunch_single(int n_frames, const float* in, const float *targ);//////////////////////////////by yongxu
	void train_bunch_single(int n_frames, float* in, const float *targ);
	//int  CrossValid(int n_frames, const float* in, const int *targ);	
	float  CrossValid(int n_frames, const float* in, const float *targ);///////////////////////////////////////by yongxu
  float  CrossValid2(int n_frames, const float* in, const float *targ);
	//void cv_bunch_single(int n_frames, const float* in, int *out);
	void cv_bunch_single(int n_frames, const float* in, float *out);///////////////////////////////////////////by yongxu
	void returnWeights(float **weights, float **bias);    			/// copy weights and biases from gpu memory to cpu memory 

	int numlayers;
	int layersizes[MAXLAYER];
	int bunchsize;
	float lrate;
	float momentum;
	float weightcost;
	int dropoutflag;
		float visible_omit;
	float hid_omit;
private:
	void devnew_vf(const char* varname, int n, float **devptr);
	void devnew_vi(const char* varname, int n, int **devptr);/////////////////////////////by yongxu
	void devfree_vf(const char* varname,  float* devptr);
	void devfree_vi(const char* varname,  int* devptr);
	void todev_vf_vf(const char* varname, int n, const float* from, float* devto, cudaStream_t stream);
	void fromdev_vf_vf(const char* varname, int n, const float* devfrom, float* to, cudaStream_t stream);
	//void todev_vi_vi(const char* varname, int n, const int* from, int* devto, cudaStream_t stream);
	//void fromdev_vi_vi(const char* varname, int n, const int* devfrom, int* to, cudaStream_t stream);

	BP_WorkSpace *dev;  //viaribles for devices
	int GPU_total;							//devices used num, ��ʾ����GPU��Ŀ
	int GPU_selected;				//devices selected, ��ʾ���õ�GPU��Ŀ

	cublasHandle_t *handles;
	cudaStream_t *streams;
		curandGenerator_t *gen;
};
