#include <vector>
#include "np.h"
#include <algorithm>
#include <omp.h>
using namespace std;

//common base class for storing layers of model in vector
class Layer
{
public:
    virtual void forward(vector<vector<double>> &input){};
    virtual void backward(vector<vector<double>> &dvalues){};
    virtual bool HasWeights(void) { return false; }
    virtual ~Layer() {}
    vector<vector<double>> output;
    vector<vector<double>> dinputs;
    vector<vector<double>> y_true;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> dweights;
    vector<double> dbiases;
    vector<vector<double>> weightmomentums;
    vector<double> biasmomentums;
    float loss;
};

//probably unneccessary
class InputLayer : public Layer
{
public:
    void forward(vector<vector<double>> &inputs)
    {
        output = inputs;
    }

    void backward(vector<vector<double>> &dvalues)
    {
        dinputs = dvalues;
    }
};

class Dense : public Layer
{
public:
    vector<vector<double>> _inputs;
    virtual bool HasWeights(void) { return true; }

    //constructor
    Dense(unsigned int n_inputs, unsigned int n_neurons)
    {
        weights = vector<vector<double>>(n_inputs, vector<double>(n_neurons));
        for (unsigned int row = 0; row < n_inputs; row++)
        {
            for (unsigned int col = 0; col < n_neurons; col++)
            {
                weights[row][col] = 0.01 * getRandomDouble(-1, 1);
            }
        }
        biases = vector<double>(n_neurons, 0);
        weightmomentums = vector<vector<double>>(n_inputs, vector<double>(n_neurons, 0));
        biasmomentums = vector<double>(n_neurons, 0);
    }

    void forward(vector<vector<double>> &inputs)
    {
        _inputs = inputs;
        output = matrixMultiply(inputs, weights);
        for (unsigned int i = 0; i < output.size(); i++)
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

class Relu : public Layer
{
public:
    vector<vector<double>> inputs;

    void forward(vector<vector<double>> &input)
    {
        inputs = input;
        output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));
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

    void backward(vector<vector<double>> &dvalues)
    {
        dinputs = vector<vector<double>>(dvalues.size(), vector<double>(dvalues[0].size(), 0));
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

class Softmax : public Layer
{
public:
    void forward(vector<vector<double>> &input)
    {
        //find max of each row
        vector<double> max(input.size(), 0);
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

        vector<vector<double>> exp_values(input.size(), vector<double>(input[0].size()));
        for (unsigned int row = 0; row < input.size(); row++)
        {
            for (unsigned int col = 0; col < input[0].size(); col++)
            {
                exp_values[row][col] = exp(input[row][col] - max[row]);
            }
        }

        vector<double> sum(input.size(), 0);
        sum = sumMatrix(exp_values, 1);

        output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));
        for (unsigned int row = 0; row < input.size(); row++)
        {
            for (unsigned int col = 0; col < input[0].size(); col++)
            {
                output[row][col] = exp_values[row][col] / sum[row];
            }
        }
    }
};

class Loss : public Layer
{
public:
    virtual vector<double> forward(vector<vector<double>> output, vector<vector<double>> y) = 0;
    float calculate(vector<vector<double>> &output, vector<vector<double>> &y)
    {
        vector<double> sample_losses = forward(output, y);
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
    vector<double> forward(vector<vector<double>> y_pred, vector<vector<double>> y_true)
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
        vector<double> sum(y_pred.size(), 0);
        sum = sumMatrix(y_pred, 1);

        //compute negative log loss of each sample
        for (unsigned int row = 0; row < sum.size(); row++)
        {
            sum[row] = -1 * log(sum[row]);
        }
        return sum;
    }
};

class SoftmaxwithLoss : public Layer
{
public:
    void forward(vector<vector<double>> &inputs)
    {
        Softmax activation;
        CategoricalCrossEntropy loss_func;
        activation.forward(inputs);
        output = activation.output;
        loss = loss_func.calculate(output, y_true);
    }

    void backward(vector<vector<double>> &dvalues)
    {
        //if one hot change to discrete value
        vector<double> discrete;
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
    float lr;
    float currentlr;
    float decay;
    int step = 0;
    float momentum = 0;
    SGD(float input_lr, float input_decay, float input_momentum) : lr{input_lr}, currentlr{input_lr}, decay{input_decay}, momentum{input_momentum} {}

    void update_params(vector<Layer *> model)
    {
        for (int layer = 0; layer < model.size(); layer++)
        {
            if (model[layer]->HasWeights())
            {
                vector<vector<double>> weight_updates(model[layer]->weights.size(), vector<double>(model[layer]->weights[0].size()));
                for (unsigned int row = 0; row < model[layer]->weights.size(); row++)
                {
                    for (unsigned int col = 0; col < model[layer]->weights[0].size(); col++)
                    {
                        weight_updates[row][col] = momentum * model[layer]->weightmomentums[row][col] + currentlr * model[layer]->dweights[row][col];
                        model[layer]->weightmomentums[row][col] = weight_updates[row][col];
                        model[layer]->weights[row][col] -= weight_updates[row][col];
                    }
                }

                vector<double> bias_updates(model[layer]->biases.size());
                for (unsigned int col = 0; col < model[layer]->dbiases.size(); col++)
                {
                    bias_updates[col] = momentum * model[layer]->biasmomentums[col] + currentlr * model[layer]->dbiases[col];
                    model[layer]->biasmomentums[col] = bias_updates[col];
                    model[layer]->biases[col] -= bias_updates[col];
                }
            }
        }
    }

    void decaylr()
    {
        currentlr = lr * (1 / (1 + decay * step));
        step += 1;
    }
};

double accuracy(vector<vector<double>> &y_pred, vector<vector<double>> &y_true)
{
    vector<double> preds = argmax(y_pred);
    vector<double> gnd_true = argmax(y_true);
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

class Model
{
public:
    vector<vector<Layer *>> ParallelModel;
    unsigned int n_threads;

    Model(vector<Layer *> input_model, unsigned int input_threads)
    {
        n_threads = input_threads;
        vector<vector<Layer *>> ParallelModel(n_threads);
        for(unsigned int thread = 0; thread < n_threads; thread++)
        {
            ParallelModel[thread] = input_model;
        }
    }

    void forward(vector<vector<double>> dataset, vector<Layer *> model)
    {
        model.front()->forward(dataset);
        for (int i = 1; i < model.size(); i++)
        {
            model[i]->forward(model[i - 1]->output);
        }
    }

    void backward(vector<vector<double>> dataset, vector<vector<double>> labels, vector<Layer *> model)
    {
        model.back()->y_true = labels;
        model.back()->backward(model.back()->output);
        for (int i = model.size() - 2; i > 0; i--)
        {
            model[i]->backward(model[i + 1]->dinputs);
        }
    }

    void train(vector<vector<vector<double>>> dataset, vector<vector<vector<double>>> labels)
    {   
        #pragma omp parallel for
        for(unsigned int thread = 0; thread < n_threads; thread++)
        {
            forward(dataset[thread], ParallelModel[thread]);
            backward(dataset[thread], labels[thread], ParallelModel[thread]);
        }
    }
};