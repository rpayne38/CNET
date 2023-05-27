#pragma once 

#include "helperFuncs.h"

class Dense
{
public:
    Matrix2d weights;
    Matrix1d biases;
    Matrix2d output;
    Matrix2d _inputs;
    Matrix2d dinputs;
    Matrix2d dweights;
    Matrix1d dbiases;
    Matrix2d weight_momentums;
    Matrix1d bias_momentums;

    //constructor
    Dense(unsigned int n_inputs, unsigned int n_neurons)
    {
        weights = Matrix2d(n_inputs, Matrix1d(n_neurons));
        for (unsigned int row = 0; row < n_inputs; row++)
        {
            for (unsigned int col = 0; col < n_neurons; col++)
            {
                weights[row][col] = 0.01 * getRandomDouble(-1, 1);
            }
        }
        biases = Matrix1d(n_neurons, 0);
        weight_momentums = Matrix2d(n_inputs, Matrix1d(n_neurons, 0));
        bias_momentums = Matrix1d(n_neurons, 0);
    }

    void forward(Matrix2d &inputs)
    {
        _inputs = inputs;
        output = matrixMultiply(inputs, weights);
        for (unsigned int i = 0; i < output.size(); i++)
        {
            output[i] = matrixAdd(output[i], biases);
        }
    }

    void backward(Matrix2d &dvalues)
    {
        Matrix2d inputs_T(_inputs[0].size(), Matrix1d(_inputs.size()));
        dweights = Matrix2d(weights.size(), Matrix1d(weights[0].size()));
        dbiases = Matrix1d(biases.size());
        Matrix2d weights_T(weights[0].size(), Matrix1d(weights.size()));
        dinputs = Matrix2d(dvalues.size(), Matrix1d(weights_T[0].size()));
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
    Matrix2d output;
    Matrix2d dinputs;
    Matrix2d inputs;

    void forward(Matrix2d &input)
    {
        inputs = input;
        output = Matrix2d(input.size(), Matrix1d(input[0].size()));
        for (unsigned int row = 0; row < input.size(); row++)
        {
            for (unsigned int col = 0; col < input[0].size(); col++)
            {
                if (input[row][col] > 0)
                {
                    output[row][col] = input[row][col];
                }
            }
        }
    }

    void backward(Matrix2d &dvalues)
    {
        dinputs = Matrix2d(dvalues.size(), Matrix1d(dvalues[0].size(), 0));
        for (unsigned int row = 0; row < dvalues.size(); row++)
        {
            for (unsigned int col = 0; col < dvalues[0].size(); col++)
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
    Matrix2d output;

    void forward(Matrix2d &input)
    {
        //find max of each row
        Matrix1d max(input.size(), 0);
        for (unsigned int row = 0; row < input.size(); row++)
        {
            for (unsigned int col = 0; col < input[0].size(); col++)
            {
                if (input[row][col] > max[row])
                {
                    max[row] = input[row][col];
                }
            }
        }

        Matrix2d exp_values(input.size(), Matrix1d(input[0].size()));
        for (unsigned int row = 0; row < input.size(); row++)
        {
            for (unsigned int col = 0; col < input[0].size(); col++)
            {
                exp_values[row][col] = exp(input[row][col] - max[row]);
            }
        }

        Matrix1d sum(input.size(), 0);
        sum = sumMatrix(exp_values, 1);

        output = Matrix2d(input.size(), Matrix1d(input[0].size()));
        for (unsigned int row = 0; row < input.size(); row++)
        {
            for (unsigned int col = 0; col < input[0].size(); col++)
            {
                output[row][col] = exp_values[row][col] / sum[row];
            }
        }
    }
};

class Loss
{
public:
    virtual Matrix1d forward(Matrix2d output, Matrix2d y) = 0;
    float calculate(Matrix2d &output, Matrix2d &y)
    {
        Matrix1d sample_losses = forward(output, y);
        float sum = 0;
        for (unsigned int i = 0; i < sample_losses.size(); i++)
        {
            sum += sample_losses[i];
        }
        return sum / sample_losses.size();
    }
};

class CategoricalCrossEntropy : public Loss
{
public:
    Matrix1d forward(Matrix2d y_pred, Matrix2d y_true)
    {
        //clip y_pred to prevent it going to infinity
        for (unsigned int row = 0; row < y_pred.size(); row++)
        {
            for (unsigned int col = 0; col < y_pred[0].size(); col++)
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
        for (unsigned int row = 0; row < y_pred.size(); row++)
        {
            for (unsigned int col = 0; col < y_pred[0].size(); col++)
            {
                y_pred[row][col] = y_pred[row][col] * y_true[row][col];
            }
        }

        //sum confidences of each sample
        Matrix1d sum(y_pred.size(), 0);
        sum = sumMatrix(y_pred, 1);

        //compute negative log loss of each sample
        for (unsigned int row = 0; row < sum.size(); row++)
        {
            sum[row] = -1 * log(sum[row]);
        }
        return sum;
    }
};

class SoftmaxwithLoss
{
public:
    Matrix2d output;
    Matrix2d dinputs;
    float loss;

    void forward(Matrix2d &inputs, Matrix2d &y_true)
    {
        Softmax activation;
        CategoricalCrossEntropy loss_func;
        activation.forward(inputs);
        output = activation.output;
        loss = loss_func.calculate(output, y_true);
    }

    void backward(Matrix2d &dvalues, Matrix2d &y_true)
    {
        //if one hot change to discrete value
        Matrix1d discrete;
        if (y_true[0].size() > 1)
        {
            discrete = argmax(y_true);
        }

        dinputs = dvalues;
        for (unsigned int row = 0; row < dinputs.size(); row++)
        {
            for (unsigned int col = 0; col < dinputs[0].size(); col++)
            {
                if (y_true[row][col] == 1)
                {
                    dinputs[row][col] -= 1;
                }
            }
        }

        int num_samples = dinputs.size();
        for (unsigned int row = 0; row < dinputs.size(); row++)
        {
            for (unsigned int col = 0; col < dinputs[0].size(); col++)
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
        Matrix2d weight_updates(A.weights.size(), Matrix1d(A.weights[0].size()));
        for (unsigned int row = 0; row < A.weights.size(); row++)
        {
            for (unsigned int col = 0; col < A.weights[0].size(); col++)
            {
                weight_updates[row][col] = _momentum * A.weight_momentums[row][col] + current_lr * A.dweights[row][col];
                A.weight_momentums[row][col] = weight_updates[row][col];
                A.weights[row][col] -= weight_updates[row][col];
            }
        }

        Matrix1d bias_updates(A.biases.size());
        for (unsigned int col = 0; col < A.dbiases.size(); col++)
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

double accuracy(Matrix2d &y_pred, Matrix2d &y_true)
{
    Matrix1d preds = argmax(y_pred);
    Matrix1d gnd_true = argmax(y_true);
    float sum = 0;

    for (unsigned int sample = 0; sample < preds.size(); sample++)
    {
        if (preds[sample] == gnd_true[sample])
        {
            sum += 1;
        }
    }
    float acc = sum / preds.size();
    return acc;
}
