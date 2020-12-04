#include <stdlib.h>
#include <iostream>
#include <immintrin.h>
#include "util.h"
using namespace std;
typedef long long int ll;

#define VEC_BYTES 32
#define MOD 2

Convolution::Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py)
{
  M = m;
  C = c;
  R = r;
  S = s;
  Sx = sx;
  Sy = sy;
  Px = px;
  Py = py;
  weights = (DATA*) malloc(M * C * R * S * sizeof(DATA));  
  DATA (*temp)[C][R][S] = (DATA (*)[C][R][S])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<R; k++)
        for(int l=0; l<S; l++)
          temp[i][j][k][l] = (i*C*R*S+j*R*S+k*S+l)%MOD;
}

Linear::Linear(int m, int l)
{
  M = m;
  L = l;
  weights = (DATA*) malloc(M * L * sizeof(DATA));
  DATA (*temp)[L] = (DATA (*)[L])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<L; j++)
      temp[i][j] = (i*L+j)%MOD;
}

void print_dims(fmap* activation){
	cout << "Activation dimensions: " << activation->dim1 << " " << activation->dim2 << " " << activation->dim3 << " " << activation->dim4 << endl;
}

void print_activation(fmap* activation){
	ll tot_size = activation->dim1 * activation->dim2 * activation->dim3 * activation->dim4;
	cout << tot_size << endl;
	for(ll i = 0;i < tot_size;i++){
		cout << activation->data[i] << " ";
	}
	cout << endl;
}

void print_4d_activation(fmap* output){
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])output->data;
	for(int b = 0;b < output->dim1;b++){
		for(int i = 0;i < output->dim2;i++){
			for(int j = 0;j < output->dim3;j++){
				for(int k = 0;k < output->dim4;k++){
					std::cout << temp_output[b][i][j][k] << " ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;std::cout << std::endl;
	}
}

void print_weights(Convolution* conv){

	DATA (*temp_output)[conv->C][conv->R][conv->S] = (DATA (*)[conv->C][conv->R][conv->S])conv->weights;
	for(int b = 0;b < conv->M;b++){
		for(int i = 0;i < conv->C;i++){
			for(int j = 0;j < conv->R;j++){
				for(int k = 0;k < conv->S;k++){
						std::cout << temp_output[b][i][j][k] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;std::cout << std::endl;
		}
	}

DATA internal_sum(__m256 x) {
    __m128 higher_half = _mm256_extractf128_ps(x, 1);
    __m128 lower_half = _mm256_extractf128_ps(x, 0);
    __m128 half_sum = _mm_add_ps(lower_half, higher_half);
    __m128 perm1 = _mm_permute_ps(half_sum, 14);
    half_sum = _mm_add_ps(half_sum, perm1);
    perm1 = _mm_permute_ps(half_sum, 1);

    // __m128 lower = half_sum;
    // __m128 higher = _mm_movehl_ps(half_sum, half_sum);
    // __m128 sum1 = _mm_add_ps(lower, higher);
    // __m128 low = sum1;
    // __m128 high = _mm_shuffle_ps(sum1, sum1, 0x1);
    __m128 sum = _mm_add_ss(half_sum, perm1);
    return (DATA)(_mm_cvtss_f32(sum));
}

fmap* Convolution::pad_input(fmap* input){
	clock_t start, end;
	start = clock();

	fmap *padded_input = new fmap();
	// keep batch and channels same
	padded_input->dim1 = input->dim1;
	padded_input->dim2 = input->dim2;
	// change ht and width
	padded_input->dim3 = input->dim3 + 2 * Px;
	padded_input->dim4 = input->dim4 + 2 * Py;
	padded_input->data = (DATA*)calloc(padded_input->dim1 * padded_input->dim2 * padded_input->dim3 * padded_input->dim4, sizeof(DATA));
	DATA (*temp_input)[input->dim2][input->dim3][input->dim4] = (DATA (*)[input->dim2][input->dim3][input->dim4])(input->data);
	DATA (*temp_padded_input)[padded_input->dim2][padded_input->dim3][padded_input->dim4] = (DATA (*)[padded_input->dim2][padded_input->dim3][padded_input->dim4])(padded_input->data);

	for(int i = 0;i < padded_input->dim1;i++){
		for(int j = 0;j < padded_input->dim2;j++){
			for(int k = Px;k < padded_input->dim3 - Px;k++){
				for(int l = Py;l < padded_input->dim4 - Py;l++){
					temp_padded_input[i][j][k][l] = temp_input[i][j][k - Px][l - Py];
				}
			}
		}
	}
	free(input->data);
	
	end = clock();
	double pad_exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	// cout << "Input pad for conv: " << pad_exec_time << endl;

	return padded_input;
}

fmap* Convolution::pad_input_fast(fmap* input){
	clock_t start, end;
	start = clock();

	int vec_size = VEC_BYTES / sizeof(DATA);

	fmap *padded_input = new fmap();
	// keep batch and channels same
	padded_input->dim1 = input->dim1;
	padded_input->dim2 = input->dim2;
	// change ht and width
	padded_input->dim3 = input->dim3 + 2 * Px;
	padded_input->dim4 = input->dim4 + 2 * Py;
	padded_input->data = (DATA*)calloc(padded_input->dim1 * padded_input->dim2 * padded_input->dim3 * padded_input->dim4, sizeof(DATA));
	DATA (*temp_input)[input->dim2][input->dim3][input->dim4] = (DATA (*)[input->dim2][input->dim3][input->dim4])(input->data);
	DATA (*temp_padded_input)[padded_input->dim2][padded_input->dim3][padded_input->dim4] = (DATA (*)[padded_input->dim2][padded_input->dim3][padded_input->dim4])(padded_input->data);

	__m256 temp;

	for(int i = 0;i < padded_input->dim1;i++){
		for(int j = 0;j < padded_input->dim2;j++){
			for(int k = Px;k < padded_input->dim3 - Px;k++){
				for(int l = Py;(l + vec_size - 1) < padded_input->dim4 - Py;l += vec_size){
					temp = _mm256_loadu_ps(&temp_input[i][j][k - Px][l - Py]);
					_mm256_storeu_ps(&temp_padded_input[i][j][k][l], temp);
					// temp_padded_input[i][j][k][l] = temp_input[i][j][k - Px][l - Py];
				}
				for(int l = ((padded_input->dim4 - Py)/vec_size)*vec_size;l < padded_input->dim4 - Py;l++)
					temp_padded_input[i][j][k][l] = temp_input[i][j][k - Px][l - Py];
			}
		}
	}
	free(input->data);
	
	end = clock();
	double pad_exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	// cout << "Input pad for conv: " << pad_exec_time << endl;

	return padded_input;
}

fmap* Convolution::conv_2d(fmap* input_features)
{
	// print_dims(input_features);
	clock_t start, end;
	start = clock();

	fmap *output = new fmap();
	output->dim1 = input_features->dim1;
	output->dim2 = this->M;
	output->dim3 = (input_features->dim3 - this->R + 2 * this->Px)/this->Sx + 1;
	output->dim4 = (input_features->dim4 - this->S + 2 * this->Py)/this->Sy + 1;

	// input_features dimensions change after this point!!!!!!!!!!!
	input_features = pad_input(input_features);
	//############################################################

	ll total_output_size = output->dim1*output->dim2*output->dim3*output->dim4;
	output->data = (DATA*)calloc(total_output_size, sizeof(DATA));

	// convert all ip, op, wts to index easily
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])(output->data);
	DATA (*temp_input)[input_features->dim2][input_features->dim3][input_features->dim4] = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])(input_features->data);
	DATA (*temp_weights)[this->C][this->R][this->S] = (DATA (*)[this->C][this->R][this->S])weights;


	for(int i_batch = 0;i_batch < output->dim1;i_batch++){
		for(int i_filter = 0;i_filter < output->dim2;i_filter++){
			for(int i_op_ht = 0;i_op_ht < output->dim3;i_op_ht++){
				for(int i_op_width = 0;i_op_width < output->dim4;i_op_width++){
					temp_output[i_batch][i_filter][i_op_ht][i_op_width] = 0;
					for(int i_ip_chn = 0;i_ip_chn < this->C;i_ip_chn++){
						for(int i_filt_height = 0; i_filt_height < this->R;i_filt_height++){
							for(int i_filt_width = 0; i_filt_width < this->S;i_filt_width++){
								temp_output[i_batch][i_filter][i_op_ht][i_op_width] += 
								temp_input[i_batch][i_ip_chn][(this->Sx*i_op_ht + i_filt_height)][(this->Sy*i_op_width + i_filt_width)] * temp_weights[i_filter][i_ip_chn][i_filt_height][i_filt_width];
							}
						}
					}
				}
			}
		}
	}
	free(input_features->data);
	end = clock();
	this->exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	double total_computations = total_output_size * this->C * this->R * this->S;
	// cout << "Conv computations: " << total_computations << endl;
	// print_dims(output);

	return output;
}

fmap* Convolution::conv2d_IS(fmap* input_features)
{
	clock_t start, end;
	start = clock();

	fmap *output = new fmap();
	output->dim1 = input_features->dim1;
	output->dim2 = this->M;
	output->dim3 = (input_features->dim3 - this->R + 2 * this->Px)/this->Sx + 1;
	output->dim4 = (input_features->dim4 - this->S + 2 * this->Py)/this->Sy + 1;

	// input_features dimensions change after this point!!!!!!!!!!!
	input_features = pad_input(input_features);
	//############################################################

	ll total_output_size = output->dim1*output->dim2*output->dim3*output->dim4;
	output->data = (DATA*)calloc(total_output_size, sizeof(DATA));
	int vec_size = VEC_BYTES / sizeof(DATA);

	// convert all ip, op, wts to index easily
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])(output->data);
	DATA (*temp_input)[input_features->dim2][input_features->dim3][input_features->dim4] = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])(input_features->data);
	DATA (*temp_weights)[this->C][this->R][this->S] = (DATA (*)[this->C][this->R][this->S])weights;

	__m256 op_reg, ip_reg, wt_reg;

	for(int i_batch = 0;i_batch < output->dim1;i_batch++){
		for(int i_ip_chn = 0;i_ip_chn < this->C;i_ip_chn++){
			for(int i_ip_height = 0; i_ip_height < input_features->dim3;i_ip_height++){
				for(int i_ip_width = 0; i_ip_width < input_features->dim4;i_ip_width++){
					DATA fixed_ip = temp_input[i_batch][i_ip_chn][i_ip_height][i_ip_width];
					ip_reg = _mm256_broadcast_ss((float*)(&fixed_ip));
					// cout << i_ip_height << " " << i_ip_width << endl;
					for(int i_filter = 0;i_filter < this->M;i_filter++){
						//######Constraints due to output size and filter size####################
						int start1 = max((i_ip_height-R)/Sx + 1, 0), start2 = max((i_ip_width-S)/Sy + 1, 0);
						int end1= min(i_ip_height/Sx, output->dim3 - 1), end2= min(i_ip_width/Sy, output->dim4-1);
						//########################################################################
						// cout << "OP_X" << start1 << "," << end1 << endl;
						// cout << "OP_Y" << start2 << "," << end2 << endl;
						
						for(int i_op_ht = start1;i_op_ht <= end1;i_op_ht++){
							for(int i_op_width = start2;(i_op_width + vec_size - 1) <= end2;i_op_width+= vec_size){
								op_reg = _mm256_loadu_ps(&temp_output[i_batch][i_filter][i_op_ht][i_op_width]);
								float *ptr = &temp_weights[i_filter][i_ip_chn][i_ip_height - this->Sx * i_op_ht][i_ip_width - this->Sy * i_op_width];
								wt_reg = _mm256_set_ps(*(ptr - 7*Sy), *(ptr - 6*Sy), *(ptr - 5*Sy), *(ptr - 4*Sy), *(ptr - 3*Sy), *(ptr - 2*Sy), *(ptr - Sy), *ptr);
								op_reg = _mm256_add_ps(op_reg, _mm256_mul_ps(ip_reg, wt_reg));
								_mm256_storeu_ps(&temp_output[i_batch][i_filter][i_op_ht][i_op_width], op_reg);
								// temp_output[i_batch][i_filter][i_op_ht][i_op_width] += 
								// fixed_ip * temp_weights[i_filter][i_ip_chn][i_ip_height - this->Sx * i_op_ht][i_ip_width - this->Sy * i_op_width];
							}
							for(int i_op_width = max(start2, (end2/ vec_size) * vec_size);i_op_width <= end2;i_op_width++){
								temp_output[i_batch][i_filter][i_op_ht][i_op_width] += 
								fixed_ip * temp_weights[i_filter][i_ip_chn][i_ip_height - this->Sx * i_op_ht][i_ip_width - this->Sy * i_op_width];	
							}
						}
					}
				}
			}
		}
	}
	free(input_features->data);
	end = clock();
	this->exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	double total_computations = total_output_size * this->C * this->R * this->S;
	// cout << "Conv computations: " << total_computations << endl;
	// print_dims(output);

	return output;
}

fmap* Convolution::conv2d_OS(fmap* input_features)
{
	clock_t start, end;
	start = clock();

	fmap *output = new fmap();
	output->dim1 = input_features->dim1;
	output->dim2 = this->M;
	output->dim3 = (input_features->dim3 - this->R + 2 * this->Px)/this->Sx + 1;
	output->dim4 = (input_features->dim4 - this->S + 2 * this->Py)/this->Sy + 1;

	// input_features dimensions change after this point!!!!!!!!!!!
	input_features = pad_input(input_features);
	//############################################################

	// Parallelized over channels

	ll total_output_size = output->dim1*output->dim2*output->dim3*output->dim4;
	output->data = (DATA*)calloc(total_output_size, sizeof(DATA));
	int vec_size = VEC_BYTES / sizeof(DATA);

	// convert all ip, op, wts to index easily
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])(output->data);	
	DATA (*temp_input)[input_features->dim2][input_features->dim3][input_features->dim4] = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])(input_features->data);
	DATA (*temp_weights)[this->C][this->R][this->S] = (DATA (*)[this->C][this->R][this->S])weights;

	__m256 acc, inp_reg, wt_reg, wt_reg2, inp_reg2;

	for(int i_batch = 0;i_batch < output->dim1;i_batch++){
		for(int i_filter = 0;i_filter < output->dim2;i_filter++){
			for(int i_op_ht = 0;i_op_ht < output->dim3;i_op_ht++){
				for(int i_op_width = 0;i_op_width < output->dim4;i_op_width++){
					DATA tmp_output = 0;
					acc = _mm256_setzero_ps();
					for(int i_ip_chn = 0;(i_ip_chn + vec_size - 1) < this->C;i_ip_chn += vec_size){
						for(int i_filt_height = 0; i_filt_height < this->R;i_filt_height++){
							for(int i_filt_width = 0; i_filt_width < this->S;i_filt_width++){
								//#######################################################################
								float *ptr = &temp_input[i_batch][i_ip_chn][(this->Sx*i_op_ht + i_filt_height)][(this->Sy*i_op_width + i_filt_width)];
								int offset = input_features->dim3 * input_features->dim4;
								inp_reg = _mm256_set_ps(*(ptr + 7*offset), *(ptr + 6*offset), *(ptr + 5*offset), *(ptr + 4*offset), *(ptr + 3*offset), *(ptr + 2*offset), *(ptr+offset), *ptr);
								
								float *ptr2 = &temp_weights[i_filter][i_ip_chn][i_filt_height][i_filt_width];
								int offset2 = this->R * this->S;
								wt_reg = _mm256_set_ps(*(ptr2 + 7*offset2), *(ptr2 + 6*offset2), *(ptr2+ 5*offset2), *(ptr2 + 4*offset2), *(ptr2 + 3*offset2), *(ptr2 + 2*offset2), *(ptr2 + offset2), *ptr2);

								acc = _mm256_add_ps(acc, _mm256_mul_ps(inp_reg, wt_reg));
								//#######################################################################
							}

						}
					}
					// corner sums left out from avx
					for(int i_filt_height = 0; i_filt_height < this->R;i_filt_height++){
						for(int i_filt_width = 0; i_filt_width < this->S;i_filt_width++){
							for(int i_ip_chn = (this->C / vec_size)*vec_size;i_ip_chn < this->C;i_ip_chn++){
								tmp_output += 
								temp_input[i_batch][i_ip_chn][(this->Sx*i_op_ht + i_filt_height)][(this->Sy*i_op_width + i_filt_width)] * temp_weights[i_filter][i_ip_chn][i_filt_height][i_filt_width];	
							}
						}
					}
					temp_output[i_batch][i_filter][i_op_ht][i_op_width] = internal_sum(acc) + tmp_output;
				}
			}
		}
	}
	free(input_features->data);
	end = clock();
	this->exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	double total_computations = total_output_size * this->C * this->R * this->S;
	// cout << "Conv computations: " << total_computations << endl;
	// print_dims(output);

	return output;
}

fmap* Convolution::conv2d_WS(fmap* input_features)
{
	clock_t start, end;
	start = clock();

	fmap *output = new fmap();
	output->dim1 = input_features->dim1;
	output->dim2 = this->M;
	output->dim3 = (input_features->dim3 - this->R + 2 * this->Px)/this->Sx + 1;
	output->dim4 = (input_features->dim4 - this->S + 2 * this->Py)/this->Sy + 1;

	// input_features dimensions change after this point!!!!!!!!!!!
	input_features = pad_input(input_features);
	//############################################################

	ll total_output_size = output->dim1*output->dim2*output->dim3*output->dim4;
	output->data = (DATA*)calloc(total_output_size, sizeof(DATA));
	int vec_size =  VEC_BYTES / sizeof(DATA);

	// convert all ip, op, wts to index easily
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])(output->data);
	DATA (*temp_input)[input_features->dim2][input_features->dim3][input_features->dim4] = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])(input_features->data);
	DATA (*temp_weights)[this->C][this->R][this->S] = (DATA (*)[this->C][this->R][this->S])weights;

	__m256 inp_reg, broadcast_wt, out_reg;
	float *ptr = NULL;
	for(int i_filter = 0;i_filter < output->dim2;i_filter++){
		for(int i_ip_chn = 0;i_ip_chn < this->C;i_ip_chn++){
			for(int i_filt_height = 0; i_filt_height < this->R;i_filt_height++){
				for(int i_filt_width = 0; i_filt_width < this->S;i_filt_width++){
					float temp_wt = temp_weights[i_filter][i_ip_chn][i_filt_height][i_filt_width];
					broadcast_wt = _mm256_broadcast_ss(&temp_wt);
					for(int i_batch = 0;i_batch < output->dim1;i_batch++){
						for(int i_op_ht = 0;i_op_ht < output->dim3;i_op_ht++){
							for(int i_op_width = 0;(i_op_width + vec_size - 1) < output->dim4;i_op_width += vec_size){
								ptr = &temp_input[i_batch][i_ip_chn][(this->Sx*i_op_ht + i_filt_height)][(this->Sy*i_op_width + i_filt_width)];
								
								inp_reg = _mm256_set_ps(*(ptr + 7*this->Sy), *(ptr + 6*this->Sy), *(ptr + 5*this->Sy), *(ptr + 4*this->Sy), *(ptr + 3*this->Sy), *(ptr + 2*this->Sy), *(ptr + this->Sy), *ptr);

								out_reg = _mm256_loadu_ps(&temp_output[i_batch][i_filter][i_op_ht][i_op_width]);
								out_reg = _mm256_add_ps(out_reg, _mm256_mul_ps(inp_reg, broadcast_wt));
								_mm256_storeu_ps(&temp_output[i_batch][i_filter][i_op_ht][i_op_width], out_reg);
							}
							for(int i_op_width = (output->dim4 / vec_size) * vec_size; i_op_width < output->dim4;i_op_width++){
								temp_output[i_batch][i_filter][i_op_ht][i_op_width] += 
								temp_input[i_batch][i_ip_chn][(this->Sx*i_op_ht + i_filt_height)][(this->Sy*i_op_width + i_filt_width)] * temp_wt;
							}
						}
					}
				}
			}
		}
	}
	free(input_features->data);
	end = clock();
	this->exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	double total_computations = total_output_size * this->C * this->R * this->S;
	// cout << "Conv computations WS: " << total_computations << endl;
	// print_dims(output);

	return output;
}

fmap* Convolution::conv2d_optimized(fmap* input_features)
{
	clock_t start, end;
	start = clock();

	fmap *output = new fmap();
	output->dim1 = input_features->dim1;
	output->dim2 = this->M;
	output->dim3 = (input_features->dim3 - this->R + 2 * this->Px)/this->Sx + 1;
	output->dim4 = (input_features->dim4 - this->S + 2 * this->Py)/this->Sy + 1;

	// input_features dimensions change after this point!!!!!!!!!!!
	input_features = pad_input(input_features);
	//############################################################

	ll total_output_size = output->dim1*output->dim2*output->dim3*output->dim4;
	output->data = (DATA*)calloc(total_output_size, sizeof(DATA));
	int vec_size =  VEC_BYTES / sizeof(DATA);

	// convert all ip, op, wts to index easily
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])(output->data);
	DATA (*temp_input)[input_features->dim2][input_features->dim3][input_features->dim4] = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])(input_features->data);
	DATA (*temp_weights)[this->C][this->R][this->S] = (DATA (*)[this->C][this->R][this->S])weights;

	__m256 inp_reg, broadcast_wt, out_reg;
	__m256i mask;
	DATA *ptr = NULL;
	// cout << "yolo" << endl;
	for(int i_filter = 0; i_filter < output->dim2;i_filter++){
		for(int i_ip_chn = 0;i_ip_chn < this->C;i_ip_chn++){
			for(int i_filt_height = 0; i_filt_height < this->R;i_filt_height++){
				for(int i_filt_width = 0; i_filt_width < this->S;i_filt_width++){
					float temp_wt = temp_weights[i_filter][i_ip_chn][i_filt_height][i_filt_width];
					broadcast_wt = _mm256_broadcast_ss(&temp_wt);
					for(int i_batch = 0;i_batch < output->dim1;i_batch++){
						int BLK1 = 1, BLK2 = 1;
						for(int i_op_ht = 0;i_op_ht < output->dim3;i_op_ht+=BLK1){
							// cout << "i_op_ht: " << i_op_ht << endl;
							for(int ii = i_op_ht;ii < min(i_op_ht + BLK1, output->dim3);ii++){
								for(int i_op_width = 0;(i_op_width + vec_size - 1) < output->dim4;i_op_width += vec_size*BLK2){
									// cout << "ii: " << ii << endl;
									for(int jj = i_op_width; (jj + vec_size - 1) < min(i_op_width + BLK2 * vec_size, output->dim4);jj++){
										ptr = &temp_input[i_batch][i_ip_chn][(this->Sx*ii + i_filt_height)][(this->Sy*jj + i_filt_width)];
										inp_reg = _mm256_set_ps(*(ptr + 7*this->Sy), *(ptr + 6*this->Sy), *(ptr + 5*this->Sy), *(ptr + 4*this->Sy), *(ptr + 3*this->Sy), *(ptr + 2*this->Sy), *(ptr + this->Sy), *ptr);
										out_reg = _mm256_loadu_ps(&temp_output[i_batch][i_filter][ii][jj]);
										out_reg = _mm256_add_ps(out_reg, _mm256_mul_ps(inp_reg, broadcast_wt));
										_mm256_storeu_ps(&temp_output[i_batch][i_filter][ii][jj], out_reg);
									}
								}
								//// Handle remaining indices:
								int idx = (output->dim4 / vec_size) * vec_size - output->dim4;
								if(idx < 0){
									ptr = &temp_output[i_batch][i_filter][ii][idx + output->dim4];
									// Way 1
									for(int i_op_width = idx + output->dim4; i_op_width < output->dim4;i_op_width++){
										*ptr += temp_input[i_batch][i_ip_chn][(this->Sx*ii + i_filt_height)][(this->Sy*i_op_width + i_filt_width)] * temp_wt;
										ptr++;
									}
								}
							}

						}
					}
				}
			}
		}
	}
	free(input_features->data);
	end = clock();
	this->exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	double total_computations = total_output_size * this->C * this->R * this->S;
	// cout << "Conv computations WS: " << total_computations << endl;
	// print_dims(output);

	return output;
}

fmap* Linear::linear(fmap* input_features)
{
	// print_dims(input_features);
	clock_t start, end;
	start = clock();
	int n = input_features->dim1, m = this->M, feats = input_features->dim2;


	fmap *output = new fmap();
	output->data = (DATA*)calloc(n * m, sizeof(DATA));
	output->dim1 = n;output->dim2 = m;output->dim3 = 1;output->dim4 = 1;

	DATA (*temp_output)[output->dim2] = (DATA (*)[output->dim2])output->data;
	DATA (*temp_input)[input_features->dim2] = (DATA (*)[input_features->dim2])input_features->data;
	DATA (*temp_weights)[this->L] = (DATA (*)[this->L])weights;

	for(int i = 0;i < n;i++){
		for(int j = 0;j < m;j++){
			for(int k = 0;k < feats;k++){
				temp_output[i][j] += temp_input[i][k] * temp_weights[j][k];
			}
		}
	}
	free(input_features->data);
	end = clock();
	this->exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	// print_dims(output);
	// cout << "Linear layer computations: " << (n * m * feats) << endl;
	return output;
}

fmap* Linear::linear_optimized(fmap* input_features)
{
	clock_t start, end;
	start = clock();
	
	ll n = input_features->dim1, m = this->M, feats = input_features->dim2;

	// Optimized in dot product of row and column
	DATA *output_map = (DATA*)calloc((n * m), sizeof(DATA));
	__m256 acc, zer;
	int vec_size = VEC_BYTES / sizeof(DATA);
	__m256 tmp1, tmp2;
	for(int i = 0;i < n;i++){
		for(int j = 0;j < m;j++){
			int op_idx = i*m + j;
			acc = _mm256_setzero_ps();
			for(int k = 0;(k + vec_size - 1) < feats;k += vec_size){
				int idx1 = i*feats + k, idx2 = j * feats + k;
				tmp1 = _mm256_loadu_ps(input_features->data + idx1);
				tmp2 = _mm256_loadu_ps(this->weights + idx2);
				acc = _mm256_add_ps(acc, _mm256_mul_ps(tmp1, tmp2));
			}

			for(int k = (feats / vec_size) * vec_size;k < feats;k++){
				output_map[op_idx] += input_features->data[i * feats + k] * this->weights[j * feats + k];
			}
			for(int k = 0;k < vec_size;k++){
				output_map[op_idx] += acc[k];
			}
		}
	}
	
	free(input_features->data);

	fmap* output = new fmap();
	output->data = output_map;
	output->dim1 = n;
	output->dim2 = m;
	output->dim3 = 1;
	output->dim4 = 1;

	end = clock();
	this->exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	// print_dims(output);

	// cout << "Linear layer computations: " << (n * m * feats) << endl;
	
	return output;
}

void relu(fmap* input_features)
{
	clock_t start, end;
	start = clock();
	ll total_size = 0;
	if(input_features != NULL){
		int vec_size = 32 / sizeof(DATA);
		total_size = input_features->dim1*input_features->dim2*input_features->dim3*input_features->dim4;
		__m256 simd_vec, zeros = _mm256_setzero_ps();
		__m256 simd_vec2;
		for(int i = 0;(i + vec_size - 1) < total_size;i += vec_size){
			simd_vec = _mm256_loadu_ps(input_features->data + i);
			simd_vec2 = _mm256_cmp_ps(simd_vec, zeros, _CMP_GE_OS);
			simd_vec = _mm256_and_ps(simd_vec, simd_vec2);
			_mm256_storeu_ps(input_features->data + i, simd_vec);
		}
		// optimize using masked load
		for(int i = (total_size / vec_size) * vec_size;i < total_size;i++){
			DATA tmp = input_features->data[i];
			input_features->data[i] = (tmp >= 0) ? tmp : 0;
		}
	}
	end = clock();
	double exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	// print_dims(input_features);
	// cout << "ReLU computes: " << total_size << endl;
	// cout << "ReLU exec time: " << exec_time << endl;
}

fmap* maxpool_2d(fmap* input_features, int R, int S, int Sx, int Sy)
{
	clock_t start, end;
	start = clock();

	fmap *output = new fmap();
	output->dim1 = input_features->dim1;
	output->dim2 = input_features->dim2;
	output->dim3 = (input_features->dim3 - R)/Sx + 1;
	output->dim4 = (input_features->dim4 - S)/Sy + 1;
	ll total_size = output->dim1*output->dim2*output->dim3*output->dim4;
	output->data = (DATA*)calloc(total_size, sizeof(DATA));

	// convert all ip, op, wts to index easily
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])(output->data);
	DATA (*temp_input)[input_features->dim2][input_features->dim3][input_features->dim4] = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])(input_features->data);

	for(int i_batch = 0;i_batch < output->dim1;i_batch++){
		for(int i_filter = 0;i_filter < input_features->dim2;i_filter++){
			for(int i_op_ht = 0;i_op_ht < output->dim3;i_op_ht++){
				for(int i_op_width = 0;i_op_width < output->dim4;i_op_width++){
					DATA tmp = -1e9;
					for(int i_filt_height = 0; i_filt_height < R;i_filt_height++){
						for(int i_filt_width = 0; i_filt_width < S;i_filt_width++){
							tmp = max(tmp, 
								temp_input[i_batch][i_filter][(Sx*i_op_ht + i_filt_height)][(Sy*i_op_width + i_filt_width)]);
						}
					}
					temp_output[i_batch][i_filter][i_op_ht][i_op_width] = tmp;
				}
			}
		}
	}
	free(input_features->data);
	end = clock();
	double exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	double total_computations = total_size * (R * S);
	// cout << "Max_pool2D computes: " << total_computations << endl;
	// cout << "Max_pool2D exec time: " << exec_time << endl;
	// print_dims(output);
	return output;
}

fmap* maxpool_2d_fast(fmap* input_features, int R, int S, int Sx, int Sy)
{
	clock_t start, end;
	start = clock();

	fmap *output = new fmap();
	output->dim1 = input_features->dim1;
	output->dim2 = input_features->dim2;
	output->dim3 = (input_features->dim3 - R)/Sx + 1;
	output->dim4 = (input_features->dim4 - S)/Sy + 1;
	ll total_size = output->dim1*output->dim2*output->dim3*output->dim4;
	output->data = (DATA*)calloc(total_size, sizeof(DATA));

	// convert all ip, op, wts to index easily
	DATA (*temp_output)[output->dim2][output->dim3][output->dim4] = (DATA (*)[output->dim2][output->dim3][output->dim4])(output->data);
	DATA (*temp_input)[input_features->dim2][input_features->dim3][input_features->dim4] = (DATA (*)[input_features->dim2][input_features->dim3][input_features->dim4])(input_features->data);

	for(int i_batch = 0;i_batch < output->dim1;i_batch++){
		for(int i_filter = 0;i_filter < input_features->dim2;i_filter++){
			for(int i_op_ht = 0;i_op_ht < output->dim3;i_op_ht++){
				for(int i_op_width = 0;i_op_width < output->dim4;i_op_width++){
					DATA tmp = -1e9;
					for(int i_filt_height = 0; i_filt_height < R;i_filt_height++){
						for(int i_filt_width = 0; i_filt_width < S;i_filt_width++){
							tmp = max(tmp, 
								temp_input[i_batch][i_filter][(Sx*i_op_ht + i_filt_height)][(Sy*i_op_width + i_filt_width)]);
						}
					}
					temp_output[i_batch][i_filter][i_op_ht][i_op_width] = tmp;
				}
			}
		}
	}
	free(input_features->data);
	end = clock();
	double exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	double total_computations = total_size * (R * S);
	// cout << "Max_pool2D computes: " << total_computations << endl;
	// cout << "Max_pool2D exec time: " << exec_time << endl;
	// print_dims(output);
	return output;
}


AlexNet::AlexNet()
{
  conv_layers = (Convolution**) malloc(5 * sizeof(Convolution*));

  Convolution *conv;
  conv = new Convolution(96, 3, 11, 11, 4, 4, 2, 2);
  conv_layers[0] = conv;
  conv = new Convolution(256, 96, 5, 5, 1, 1, 2, 2);
  conv_layers[1] = conv;
  conv = new Convolution(384, 256, 3, 3, 1, 1, 1, 1);
  conv_layers[2] = conv;
  conv = new Convolution(384, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[3] = conv;
  conv = new Convolution(256, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[4] = conv;

  linear_layers = (Linear**) malloc(3 * sizeof(Linear*));

  Linear *linear;
  linear = new Linear(4096, 9216);
  linear_layers[0] = linear;
  linear = new Linear(4096, 4096);
  linear_layers[1] = linear;
  linear = new Linear(1000, 4096);
  linear_layers[2] = linear;
}

fmap* AlexNet::test_forward_pass(fmap *input_features, int kk){
	clock_t start, end;
	start = clock();

	fmap* temp = input_features;
	
	Convolution *conv, *conv2;
	Linear *lin1;
	conv = new Convolution(1, 1, 2, 2, 1, 1, 1, 1);
	// conv2 = new Convolution(2, 1, 3, 3, 1, 1, 0, 0);
	// lin1 = new Linear(2, 12);
	print_weights(conv);
	// print_weights(conv2);
	// print_weights(Convolution *conv)
	temp = conv->conv2d_IS(temp);
	// temp = maxpool_2d(temp, 2, 2, 2, 2);
	// temp = conv2->conv_2d(temp);
	// temp = lin1->linear_optimized(temp);


	end = clock();

	exec_time = double(end-start) / double(CLOCKS_PER_SEC);

	return temp;	
}

fmap* AlexNet::forward_pass(fmap* input_features, int dataflow)
{
	clock_t start, end;
	start = clock();

	fmap* temp = input_features;
	if(dataflow == 0)temp = conv_layers[0]->conv2d_IS(temp);
	else if(dataflow == 1)temp = conv_layers[0]->conv2d_WS(temp);
	else if(dataflow == 2)temp = conv_layers[0]->conv2d_OS(temp);
	else temp = conv_layers[0]->conv2d_optimized(temp);
	relu(temp);
	temp = maxpool_2d(temp, 3, 3, 2, 2);
	if(dataflow == 0)temp = conv_layers[1]->conv2d_IS(temp);
	else if(dataflow == 1)temp = conv_layers[1]->conv2d_WS(temp);
	else if(dataflow == 2)temp = conv_layers[1]->conv2d_OS(temp);
	else temp = conv_layers[1]->conv2d_optimized(temp);
	relu(temp);
	temp = maxpool_2d(temp, 3, 3, 2, 2);
	if(dataflow == 0)temp = conv_layers[2]->conv2d_IS(temp);
	else if(dataflow == 1)temp = conv_layers[2]->conv2d_WS(temp);
	else if(dataflow == 2)temp = conv_layers[2]->conv2d_OS(temp);
	else temp = conv_layers[2]->conv2d_optimized(temp);
	// temp = conv_layers[2]->conv2d_WS(temp);
	relu(temp);
	if(dataflow == 0)temp = conv_layers[3]->conv2d_IS(temp);
	else if(dataflow == 1)temp = conv_layers[3]->conv2d_WS(temp);
	else if(dataflow == 2)temp = conv_layers[3]->conv2d_OS(temp);
	else temp = conv_layers[3]->conv2d_optimized(temp);
	// temp = conv_layers[3]->conv2d_WS(temp);
	relu(temp);
	if(dataflow == 0)temp = conv_layers[4]->conv2d_IS(temp);
	else if(dataflow == 1)temp = conv_layers[4]->conv2d_WS(temp);
	else if(dataflow == 2)temp = conv_layers[4]->conv2d_OS(temp);
	else temp = conv_layers[4]->conv2d_optimized(temp);
	// temp = conv_layers[4]->conv2d_WS(temp);
	relu(temp);
	temp = maxpool_2d(temp, 3, 3, 2, 2);

	int lin_dim = temp->dim2 * temp->dim3 * temp->dim4;
	temp->dim2 = lin_dim;
	temp->dim3 = temp->dim4 = 1;

	temp = linear_layers[0]->linear_optimized(temp);
	relu(temp);
	temp = linear_layers[1]->linear_optimized(temp);
	relu(temp);
	temp = linear_layers[2]->linear_optimized(temp);
	relu(temp);

	end = clock();

	exec_time = double(end-start) / double(CLOCKS_PER_SEC);
	return temp;
}