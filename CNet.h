#include <iostream>
#include <vector>
#include <random>
#include "np.h"
using namespace std;

class Dense
{
public:
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> output;
    //TODO fix this awfulness
    vector<vector<double>> _inputs;
    vector<vector<double>> dinputs;
    vector<vector<double>> dweights;
    vector<double> dbiases;
    vector<vector<double>> weight_momentums;
    vector<double> bias_momentums;

    //constructor
    Dense(unsigned int n_inputs, unsigned int n_neurons)
    {
        weights = vector<vector<double>>(n_inputs, vector<double>(n_neurons));
        #pragma omp parallel for collapse(2) 
        for (int row = 0; row < n_inputs; row++)
        {
            for (int col = 0; col < n_neurons; col++)
            {
                weights[row][col] = 0.01 * getRandomDouble(-1, 1);
            }
        }
        biases = vector<double>(n_neurons, 0);
        weight_momentums = vector<vector<double>>(n_inputs, vector<double>(n_neurons, 0));
        bias_momentums = vector<double>(n_neurons, 0);
        
    }

    void forward(vector<vector<double>> &inputs)
    {
        _inputs = inputs;
        output = matrixMultiply(inputs, weights);
        for (int i = 0; i < output.size(); i++)
        {
            output[i] = matrixAdd(output[i], biases);
        }
    }

    void backward(vector<vector<double>> &dvalues)
    {
        vector<vector<double>> inputs_T(_inputs[0].size(), vector<double>(_inputs.size()));
        dweights = vector<vector<double>>(weights.size(), vector<double>(weights[0].size()));
        dbiases = vector<double>(biases.size());
        vector<vector<double>> weights_T(weights[0].size(), vector<double>(weights.size()));
        dinputs = vector<vector<double>>(dvalues.size(), vector<double>(weights_T[0].size()));
        inputs_T = transpose(_inputs);
        dweights = matrixMultiply(inputs_T, dvalues);
        dbiases = sumMatrix(dvalues, 0);
        weights_T = transpose(weights);
        dinputs = matrixMultiply(dvalues, weights_T);
    }
};

class Relu
{
public:
    vector<vector<double>> output;
    vector<vector<double>> dinputs;
    vector<vector<double>> inputs;

    void forward(vector<vector<double>> &input)
    {
        inputs = input;
        output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));
        #pragma omp parallel for collapse(2) 
        for (int row = 0; row < input.size(); row++)
        {
            for (int col = 0; col < input[0].size(); col++)
            {
                if (input[row][col] > 0)
                {
                    output[row][col] = input[row][col];
                }
            }
        }
    }

    void backward(vector<vector<double>> &dvalues)
    {
        dinputs = vector<vector<double>>(dvalues.size(), vector<double>(dvalues[0].size(), 0));
        #pragma omp parallel for collapse(2) 
        for (int row = 0; row < dvalues.size(); row++)
        {
            for (int col = 0; col < dvalues[0].size(); col++)
            {
                if (inputs[row][col] > 0)
                {
                    dinputs[row][col] = dvalues[row][col];
                }
            }
        }
    }
};

class Softmax
{
public:
    vector<vector<double>> output;

    void forward(vector<vector<double>> &input)
    {
        //find max of each row
        vector<double> max(input.size(), 0);
        for (int row = 0; row < input.size(); row++)
        {
            for (int col = 0; col < input[0].size(); col++)
            {
                if (input[row][col] > max[row])
                {
                    max[row] = input[row][col];
                }
            }
        }

        vector<vector<double>> exp_values(input.size(), vector<double>(input[0].size()));
        for (int row = 0; row < input.size(); row++)
        {
            for (int col = 0; col < input[0].size(); col++)
            {
                exp_values[row][col] = exp(input[row][col] - max[row]);
            }
        }

        vector<double> sum(input.size(), 0);
        sum = sumMatrix(exp_values, 1);

        output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));
        for (int row = 0; row < input.size(); row++)
        {
            for (int col = 0; col < input[0].size(); col++)
            {
                output[row][col] = exp_values[row][col] / sum[row];
            }
        }
    }
};

class Loss
{
public:
    virtual vector<double> forward(vector<vector<double>> output, vector<vector<double>> y){};
    float calculate(vector<vector<double>> &output, vector<vector<double>> &y)
    {
        vector<double> sample_losses = forward(output, y);
        float sum = 0;
        for (int i = 0; i < sample_losses.size(); i++)
        {
            sum += sample_losses[i];
        }
        return sum / sample_losses.size();
    }
};

class CategoricalCrossEntropy : public Loss
{
public:
    vector<double> forward(vector<vector<double>> y_pred, vector<vector<double>> y_true)
    {
        //clip y_pred to prevent it going to infinity
        #pragma omp parallel for collapse(2) 
        for (int row = 0; row < y_pred.size(); row++)
        {
            for (int col = 0; col < y_pred[0].size(); col++)
            {
                if (y_pred[row][col] < 1e-7)
                {
                    y_pred[row][col] = 1e-7;
                }
                else if (y_pred[row][col] > 1 - 1e-7)
                {
                    y_pred[row][col] = 1 - 1e-7;
                }
            }
        }

        //multiply each prediction by each y_true
        for (int row = 0; row < y_pred.size(); row++)
        {
            for (int col = 0; col < y_pred[0].size(); col++)
            {
                y_pred[row][col] = y_pred[row][col] * y_true[row][col];
            }
        }

        //sum confidences of each sample
        vector<double> sum(y_pred.size(), 0);
        sum = sumMatrix(y_pred, 1);

        //compute negative log loss of each sample
        #pragma omp parallel for 
        for (int row = 0; row < sum.size(); row++)
        {
            sum[row] = -1 * log(sum[row]);
        }
        return sum;
    }
};

class SoftmaxwithLoss
{
public:
    vector<vector<double>> output;
    vector<vector<double>> dinputs;
    float loss;

    void forward(vector<vector<double>> &inputs, vector<vector<double>> &y_true)
    {
        Softmax activation;
        CategoricalCrossEntropy loss_func;
        activation.forward(inputs);
        output = activation.output;
        loss = loss_func.calculate(output, y_true);
    }

    void backward(vector<vector<double>> &dvalues, vector<vector<double>> &y_true)
    {
        //if one hot change to discrete value
        vector<double> discrete;
        if (y_true[0].size() > 1)
        {
            discrete = argmax(y_true);
        }

        dinputs = dvalues;
        #pragma omp parallel for collapse(2) 
        for (int row = 0; row < dinputs.size(); row++)
        {
            for (int col = 0; col < dinputs[0].size(); col++)
            {
                if (y_true[row][col] == 1)
                {
                    dinputs[row][col] -= 1;
                }
            }
        }

        int num_samples = dinputs.size();
        for (int row = 0; row < dinputs.size(); row++)
        {
            for (int col = 0; col < dinputs[0].size(); col++)
            {
                dinputs[row][col] /= num_samples;
            }
        }
    }
};

class SGD
{
public:
    float _lr;
    float current_lr;
    float _decay;
    int _step = 0;
    float _momentum = 0;
    SGD(float lr, float decay, float momentum)
    {
        _lr = lr;
        current_lr = lr;
        _decay = decay;
        _momentum = momentum;
    }

    void update_params(Dense &A)
    {
        vector<vector<double>> weight_updates(A.weights.size(), vector<double>(A.weights[0].size()));
        for (int row = 0; row < A.weights.size(); row++)
        {
            for (int col = 0; col < A.weights[0].size(); col++)
            {
                weight_updates[row][col] = _momentum * A.weight_momentums[row][col] + current_lr * A.dweights[row][col];
                A.weight_momentums[row][col] = weight_updates[row][col];
                A.weights[row][col] -= weight_updates[row][col];
            }
        }

        vector<double> bias_updates(A.biases.size());
        for (int col = 0; col < A.dbiases.size(); col++)
        {
            bias_updates[col] = _momentum * A.bias_momentums[col] + current_lr * A.dbiases[col];
            A.bias_momentums[col] = bias_updates[col];
            A.biases[col] -= bias_updates[col];
        }
    }

    void decay_lr()
    {
        current_lr = _lr * (1 / (1 + _decay * _step));
        _step += 1;
    }
};


double accuracy(vector<vector<double>> &y_pred, vector<vector<double>> &y_true)
{
    vector<double> preds = argmax(y_pred);
    vector<double> gnd_true = argmax(y_true);
    float sum = 0;

    for(int sample = 0; sample < preds.size(); sample++)
    {
        if(preds[sample] == gnd_true[sample])
        {
            sum += 1;
        }
    }
    float acc = sum / preds.size();
    return acc;
}
