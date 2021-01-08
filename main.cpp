//import necessary libraries
#include <fstream>
#include "CNet.h"
#include <sys/time.h>
using namespace std;

int main()
{
    struct timeval start, end;
    gettimeofday(&start, NULL);

    cout << "Loading data...\n";
    vector<vector<double>> dataset = read_mnist_imgs("train-images.idx3-ubyte");
    vector<vector<double>> labels = read_mnist_labels("train-labels.idx1-ubyte");

    cout << "Loading model...\n";
    //declare model
    Dense Dense1(28 * 28, 32);
    Relu Activation1;
    Dense Dense2(32, 32);
    Relu Activation2;
    Dense Dense3(32, 10);
    SoftmaxwithLoss softmax;
    SGD optimizer(0.01, 0.05, 0.9);

    int epochs = 20;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        cout << "Epoch: " << epoch + 1 << "\n";
        //forward pass
        Dense1.forward(dataset);
        Activation1.forward(Dense1.output);
        Dense2.forward(Activation1.output);
        Activation2.forward(Dense2.output);
        Dense3.forward(Activation2.output);
        softmax.forward(Dense3.output, labels);

        //output of model
        cout << "Loss: " << softmax.loss << "\tAccuracy: " << accuracy(softmax.output, labels) << "\t"
             << "Lr: " << optimizer.current_lr << "\n";

        //backward pass
        softmax.backward(softmax.output, labels);
        Dense3.backward(softmax.dinputs);
        Activation2.backward(Dense3.dinputs);
        Dense2.backward(Activation2.dinputs);
        Activation1.backward(Dense2.dinputs);
        Dense1.backward(Activation1.dinputs);

        //update params
        optimizer.update_params(Dense1);
        optimizer.update_params(Dense2);
        optimizer.update_params(Dense3);
        optimizer.decay_lr();
    }
    cout << "\n";

    gettimeofday(&end, NULL);
    float delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    cout << delta << "secs"
         << "\n";
    cout << delta / epochs << " secs/epoch"
         << "\n";
    return 0;
}
