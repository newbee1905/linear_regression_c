#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ARR_LEN(xs) (sizeof((xs)) / sizeof((xs)[0]))
#define EPS 1e-5

typedef float (*act_fn_t)(float);
typedef float (*loss_fn_t)(float *, float *, size_t);

inline float randf(void);
inline float rmse(float* y_hat, float *y_true, size_t sz);
inline float sigmoidf(float x);
inline float reluf(float x);
inline float nonef(float x);

typedef struct {
	size_t inp_sz;
	size_t out_sz;

	float *w;
	float *dw;

	float *out;
} layer_t;

typedef struct {
	layer_t** layers;

	size_t sz;
} nn_t;

layer_t* layer_alloc(size_t inp_sz, size_t out_sz);
void layer_free(layer_t* layer);

nn_t* nn_alloc(layer_t** layers, size_t sz);
void nn_free(nn_t* nn);

void forward_layer(layer_t* layer, float *inp, act_fn_t act);

void forward_nn(nn_t* nn, float *inp, act_fn_t act);
void backward_nn(nn_t* nn, act_fn_t act, float *X_train, float *y_train, float* y_hat, size_t train_sz, size_t features_sz, float loss, loss_fn_t loss_fn, float lr);

void print_layer(layer_t* layer);
void print_nn(nn_t* nn);

float train[][7] = {
	{2, 3, 3, 2, 10},
	{1, 1.5, 1.5, 1, 5},
	{1, 2, 1, 1.2, 5.2},
	{1, 1.4, 2, 1.3, 5.7},
	{1, 1, 1.8, 2.7, 6.5},
	{1, 2.9, 2.22, 3, 9.11},
	{10, 10.789, 10, 10, 40.789},
	{19, 21.456, 12.123, 15, 67.579},
};

float* test_gen(size_t sz, size_t features_sz);

int main() {
	srand(time(NULL));

	float lr = 1e-3;
	size_t epoch_nums = 5e4;

	size_t train_sz = ARR_LEN(train) * 0.67;
	size_t test_sz = ARR_LEN(train) - train_sz;
	size_t features_sz = 4;

	float* X_train = malloc((train_sz * features_sz) * sizeof(*X_train));
	float* y_train = malloc(train_sz * sizeof(*y_train));

	float* X_test = malloc((test_sz * features_sz) * sizeof(*X_train));
	float* y_test = malloc(test_sz * sizeof(*y_train));

	float* y_hat = malloc(train_sz * sizeof(*y_train));
	float* y_hat_e = malloc(train_sz * sizeof(*y_train));
	float* y_hat_test = malloc(test_sz * sizeof(*y_train));

	for (size_t i = 0; i < train_sz; ++i) {
		for (size_t j = 0; j < features_sz; ++j) {
			X_train[i * features_sz + j] = train[i][j];
		}
		y_train[i] = train[i][features_sz];
	}

	for (size_t i = 0; i < test_sz; ++i) {
		for (size_t j = 0; j < features_sz; ++j) {
			X_test[i * features_sz + j] = train[i + train_sz][j];
		}
		y_test[i] = train[i + train_sz][features_sz];
	}

	size_t layers_sz = 3;
	layer_t** layers = malloc(layers_sz * sizeof(layer_t*));

	layers[0] = layer_alloc(4, 6);
	layers[1] = layer_alloc(6, 2);
	layers[2] = layer_alloc(2, 1);

	nn_t* nn = nn_alloc(layers, layers_sz);

	float loss;
	for (size_t e = 0; e < epoch_nums; ++e) {
		// training
		for (size_t i = 0; i < train_sz; ++i) {
			forward_nn(nn, X_train + i * features_sz, reluf);
			y_hat[i] = nn->layers[nn->sz - 1]->out[0];
		}
		loss = rmse(y_hat, y_train, train_sz);

		backward_nn(nn, reluf, X_train, y_train, y_hat_e, train_sz, features_sz, loss, rmse, lr);

		// validating
		for (size_t i = 0; i < test_sz; ++i) {
			forward_nn(nn, X_test + i * features_sz, reluf);
			y_hat_test[i] = nn->layers[nn->sz - 1]->out[0];
		}
		loss = rmse(y_hat_test, y_test, train_sz);
		printf("Epoch %u: %f\n", e, loss);
	}
	
	for (size_t i = 0; i < train_sz; ++i) {
		for (size_t j = 0; j < features_sz; ++j) {
			printf("%f ", X_train[i * features_sz + j]);
		}
		printf("(%f) -> ", y_train[i]);
		printf("%f\n", y_hat[i]);
	}
	for (size_t i = 0; i < test_sz; ++i) {
		for (size_t j = 0; j < features_sz; ++j) {
			printf("%f ", X_test[i * features_sz + j]);
		}
		printf("(%f) -> ", y_test[i]);
		printf("%f\n", y_hat_test[i]);
	}
	printf("\n");

	printf("Testing...\n");

	float* X_true = test_gen(100, features_sz);
	float* y_true = malloc(sizeof(float) * 100);
	float* y_hat_true = malloc(sizeof(float) * 100);

	for (size_t i = 0; i < 100; ++i) {
		forward_nn(nn, X_true + i * features_sz, reluf);
		y_true[i] = 0;
		for (size_t j = 0; j < features_sz; ++j) {
			y_true[i] += X_true[i * features_sz + j];
		}
		y_hat_true[i] = nn->layers[nn->sz - 1]->out[0];
	}
	loss = rmse(y_hat_true, y_true, 100);
	printf("LOSS: %f\n", loss);

	nn_free(nn);
	free(layers);
	free(X_train);
	free(y_train);
	free(X_test);
	free(y_test);
	free(y_hat);
	free(y_hat_e);
	free(y_hat_test);
	free(X_true);
	free(y_true);
	free(y_hat_true);
}

float randf(void) {
	return (float)rand() / (float)RAND_MAX;
}

float rmse(float* y_hat, float *y_true, size_t sz) {
	float loss = 0;
	for (size_t i = 0; i < sz; ++i) {
		loss += (y_hat[i] - y_true[i]) * (y_hat[i] - y_true[i]);
	}

	return sqrt(loss / (sz * 1.0f));
}

float sigmoidf(float x) {
	return 1.f / (1.f + expf(-x));
}

float reluf(float x) {
	return (x > 0) ? x : 0.0f;
}

float nonef(float x) {
	return x;
}


layer_t* layer_alloc(size_t inp_sz, size_t out_sz) {
	layer_t* layer = malloc(sizeof(layer_t));

	layer->inp_sz = inp_sz;
	layer->out_sz = out_sz;

	layer->w = malloc(sizeof(float) * (inp_sz + 1));
	layer->dw = malloc(sizeof(float) * (inp_sz + 1));

	layer->out = malloc(sizeof(float) * out_sz);

	for (int i = 0; i <= layer->inp_sz; layer->w[i++] = randf());

	return layer;
}

void layer_free(layer_t* layer) {
	free(layer->w);
	free(layer->dw);
	free(layer->out);
	free(layer);
}

nn_t* nn_alloc(layer_t** layers, size_t sz) {
	nn_t* nn = malloc(sizeof(nn_t));
	nn->layers = layers;
	nn->sz = sz;

	return nn;
}

void nn_free(nn_t* nn) {
	for (size_t i = 0; i < nn->sz; ++i) {
		layer_free(nn->layers[i]);
	}
	free(nn);
}

void forward_layer(layer_t* layer, float *inp, act_fn_t act) {
	for (size_t i = 0; i < layer->out_sz; ++i) {
		layer->out[i] = layer->w[0];

		for (size_t j = 0; j < layer->inp_sz; ++j) {
			layer->out[i] += layer->w[j + 1] * inp[j];
		}

		layer->out[i] = (*act)(layer->out[i]);
	}
}


void forward_nn(nn_t* nn, float *inp, act_fn_t act) {
	forward_layer(nn->layers[0], inp, act);
	for (size_t i = 1; i < nn->sz; ++i) {
		forward_layer(nn->layers[i], nn->layers[i - 1]->out, act);
	}
}

void backward_nn(nn_t* nn, act_fn_t act, float *X_train, float *y_train, float* y_hat, size_t train_sz, size_t features_sz, float loss, loss_fn_t loss_fn, float lr) {
	for (size_t l = 0; l < nn->sz; ++l) {
		for (size_t w = 0; w <= nn->layers[l]->inp_sz; ++w) {
			nn->layers[l]->w[w] += EPS;
			for (size_t i = 0; i < train_sz; ++i) {
				forward_nn(nn, X_train + i * features_sz, act);
				y_hat[i] = nn->layers[nn->sz - 1]->out[0];
			}
			nn->layers[l]->w[w] -= EPS;


			float loss_e = (*loss_fn)(y_hat, y_train, train_sz);
			nn->layers[l]->dw[w] = (loss_e - loss) / EPS;
		}
	}

	for (size_t i = 0; i < nn->sz; ++i) {
		for (size_t j = 0; j <= nn->layers[i]->inp_sz; ++j) {
			nn->layers[i]->w[j] -= nn->layers[i]->dw[j] * lr;
		}
	}
}

void print_layer(layer_t* layer) {
	for (size_t i = 0; i < layer->out_sz; ++i) {
		printf("%f ", layer->out[i]);
	}
	printf("\n");
}

void print_nn(nn_t* nn) {
	for (size_t i = 0; i < nn->sz; ++i) {
		print_layer(nn->layers[i]);
	}
	printf("\n");
}

float* test_gen(size_t sz, size_t features_sz) {
	float* test = malloc(sizeof(float) * sz * features_sz);
	for (size_t i = 0; i < sz * features_sz; ++i) {
		test[i] = randf();
	}

	return test;
}
