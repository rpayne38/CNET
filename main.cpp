//import necessary libraries
#include <iostream>
#include <fstream>
#include <vector>
#include "CNet.h"
#include <sys/time.h>
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
    vector<vector<double>> dataset;
    ifstream file (path, ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        unsigned int number_of_images=0;
        unsigned int n_rows=0;
        unsigned int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        dataset = vector<vector<double>>(number_of_images, vector<double>(n_rows*n_cols));
        for(unsigned int i=0;i<number_of_images;++i)
        {
            for(unsigned int r=0;r<n_rows;++r)
            {
                for(unsigned int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    dataset[i][(28*r)+c] = temp;
                }
            }
        }
    }
    file.close();
    return dataset;
}

vector<vector<double>> read_mnist_labels(string path)
{
    vector<vector<double>> dataset;
    ifstream file (path, ios::binary);
    if (file.is_open())
    {
        unsigned int labels = 10;
        int magic_number=0;
        unsigned int number_of_labels=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);
        dataset = vector<vector<double>>(number_of_labels, vector<double>(labels, 0));
        for(unsigned int i=0;i<number_of_labels;++i)
        {
            {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                dataset[i][temp] = 1;
            }
        }
    }
    file.close();
    return dataset;
}

int main()
{
    struct timeval start, end;
    gettimeofday(&start, NULL);

    cout << "Loading data...\n";
    vector<vector<double>> dataset = read_mnist_imgs("train-images.idx3-ubyte");
    vector<vector<double>> labels = read_mnist_labels("train-labels.idx1-ubyte");

    cout << "Loading model...\n";
     
    //declare model
    Layer *model[7];
    InputLayer *Input = new InputLayer();
    Dense *Dense1 = new Dense(28*28, 32);
    Relu *Activation1 = new Relu();
    Dense *Dense2 = new Dense(32, 32);
    Relu *Activation2 = new Relu();
    Dense *Dense3 = new Dense(32, 10);
    SoftmaxwithLoss *softmax = new SoftmaxwithLoss();
    SGD optimizer(0.01, 0.05, 0.9);

    model[0] = Input;
    model[1] = Dense1;
    model[2] = Activation1;
    model[3] = Dense2;
    model[4] = Activation2;
    model[5] = Dense3;
    model[6] = softmax;
    model[6] -> y_true = labels;

    int epochs = 50;

    for(int epoch = 0; epoch < epochs; epoch++)
    {   
        cout << "Epoch: " << epoch + 1 << "\n";

        //forward pass
        Input -> forward(dataset);
        for(int i = 1; i < 7; i++)
        {
            model[i] -> forward(model[i-1] -> output);
        }

        cout << "Loss: " << softmax -> loss << "\tAccuracy: " << accuracy(softmax -> output, labels) << "\t" << "Lr: " << optimizer.current_lr << "\n";

        //backward pass
        softmax -> backward(softmax -> output);
        for(int i = 5; i > 0; i--)
        {
            model[i] -> backward(model[i+1] -> dinputs);
        }

        //update params
        optimizer.update_params(Dense1);
        optimizer.update_params(Dense2);
        optimizer.update_params(Dense3);
        optimizer.decay_lr();
    }

    cout << "\n";

    gettimeofday(&end, NULL);
    float delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    cout << delta << "secs";
    cout << "\n";

    return 0;
}
