//import necessary libraries
#include <iostream>
#include <fstream>
#include <vector>
#include "CNet.h"
#include <sys/time.h>
using namespace std;

int main()
{
    //start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    cout << "Loading data...\n";
    unsigned int n_threads = 4;
    unsigned int epochs = 10;
    vector<vector<vector<double>>> dataset = read_mnist_imgs("train-images.idx3-ubyte", n_threads);
    vector<vector<vector<double>>> labels = read_mnist_labels("train-labels.idx1-ubyte", n_threads);

    cout << "Loading model...\n";

    //declare model
    vector<Layer *> Master;
    Master.push_back(new InputLayer());
    Master.push_back(new Dense(28 * 28, 32));
    Master.push_back(new Relu());
    Master.push_back(new Dense(32, 32));
    Master.push_back(new Relu());
    Master.push_back(new Dense(32, 10));
    Master.push_back(new SoftmaxwithLoss());
    SGD optimizer(0.01, 0.05, 0.9);

    Model model(Master, n_threads);
    model.train(dataset, labels);

    cout << "\n";
    //end timer
    gettimeofday(&end, NULL);
    float delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    cout << delta << "secs"
         << "\n";
    cout << delta / epochs << " secs/epoch";
    cout << "\n";

    return 0;
}
