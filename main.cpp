//import necessary libraries
#include <iostream>
#include <fstream>
#include <vector>
#include "CNet.h"
#include <ctime>
using namespace std;

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
vector<vector<double>> read_mnist_imgs(string path)
{
    ifstream file (path, ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        vector<vector<double>> dataset(number_of_images, vector<double>(n_rows*n_cols));
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    dataset[i][(28*r)+c] = temp;
                }
            }
        }
        return dataset;
    }
    file.close();
}

vector<vector<double>> read_mnist_labels(string path)
{
    ifstream file (path, ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_labels=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);
        vector<vector<double>> dataset(number_of_labels, vector<double>(10, 0));
        for(int i=0;i<number_of_labels;++i)
        {
            {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                dataset[i][temp] = 1;
            }
        }
        return dataset;
    }
    file.close();
}

int main()
{
    clock_t start;
    start = clock();

    cout << "Loading data...\n";
    vector<vector<double>> dataset = read_mnist_imgs("/home/rob/Documents/c++/train-images.idx3-ubyte");
    vector<vector<double>> labels = read_mnist_labels("/home/rob/Documents/c++/train-labels.idx1-ubyte");

    cout << "Loading model...\n";
    //declare model
    Dense Dense1(28*28, 32);
    Relu Activation1;
    Dense Dense2(32, 64);
    Relu Activation2;
    Dense Dense3(64, 10);
    SoftmaxwithLoss softmax;
    SGD optimizer(0.02, 0.1, 0.9);

    int epochs = 10;

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
        cout << "Loss: " << softmax.loss << "\tAccuracy: " << accuracy(softmax.output, labels) << "\t" << "Lr: " << optimizer.current_lr << "\n";

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

    double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << duration << "secs";
    cout << "\n";
}