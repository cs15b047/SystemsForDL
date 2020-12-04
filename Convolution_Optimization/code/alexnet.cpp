#include <bits/stdc++.h>
#include <stdlib.h>
#include "util.h"

#define N 1
#define C 3
#define H 227
#define W 227

// #define N 1
// #define C 1
// #define H 5
// #define W 5
using namespace std;

fmap populate_input(){
	fmap input;
	input.dim1 = N;
	input.dim2 = C;
	input.dim3 = H;
	input.dim4 = W;
	input.data = (DATA*) malloc(N * C * H * W * sizeof(DATA));

	DATA (*temp)[C][H][W] = (DATA (*)[C][H][W])input.data;

	for(int i=0; i<N; i++)
	for(int j=0; j<C; j++)
	  for(int k=0; k<H; k++)
	    for(int l=0; l<W; l++)
	      temp[i][j][k][l] = (DATA)((i*C*H*W+j*H*W+k*W+l)%256);
	return input;
}

int main()
{
	AlexNet net;

	int num_inferences = 20;
	vector<vector<double> > inference_times(5, vector<double>(5, 0));
	int k;
	cin >> k;
	for(int j = 0;j < num_inferences;j++){
		fmap input = populate_input();
		// print_4d_activation(&input);
		fmap* output = net.forward_pass(&input, k);
		// print_4d_activation(output);
		
		// cout << endl;
		// exit(0);

		for(int i=0; i<5; i++){
			// cout << net.conv_layers[i]->exec_time << " ";
			inference_times[i][k] += net.conv_layers[i]->exec_time;
		}

		// for(int i=0; i<3; i++)
		//   cout << net.linear_layers[i]->exec_time << " ";

		cout << net.exec_time << endl;
	}
	for(int i=0; i<5; i++){
		inference_times[i][k] /= (double)num_inferences;
	}
	if(k == 0)cout << "IS:" << endl;
	else if(k == 1)cout << "WS:" << endl;
	else if(k == 2)cout << "OS:" << endl;
	for(int i = 0; i < 5;i++){
		cout << inference_times[i][k] << " ";
	}
	cout << endl;

  return 0;
}
