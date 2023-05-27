#include "CNet.h"

#ifdef _WIN32
#include <chrono>
#else
#include <sys/time.h>
#endif

using namespace std;

int NUM_EPOCHS = 5;
int BATCH_SIZE = 1000;

int main()
{
    //Start timer
#ifdef _WIN32
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#else
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif

    cout << "Loading data...\n";
    vector<vector<double>> dataset = read_mnist_imgs("train-images.idx3-ubyte");
    vector<vector<double>> labels = read_mnist_labels("train-labels.idx1-ubyte");
    const int NUM_BATCHES = dataset.size() / BATCH_SIZE;

    cout << "Loading model...\n";
    Dense Dense1(28 * 28, 32);
    Relu Activation1;
    Dense Dense2(32, 32);
    Relu Activation2;
    Dense Dense3(32, 10);
    SoftmaxwithLoss softmax;
    SGD optimizer(0.01, 0.05, 0.9);

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        vector<double> acc(BATCH_SIZE);
        vector<double> loss(BATCH_SIZE);

        cout << "Epoch: " << epoch + 1 << "\n";
        for (int step = 0; step < dataset.size() / BATCH_SIZE; step++)
        {
            //Extract batch from dataset
            vector<vector<double>> IMG_BATCH(BATCH_SIZE, vector<double>(dataset[0].size()));
            vector<vector<double>> LABEL_BATCH(BATCH_SIZE, vector<double>(labels[0].size()));
            batch_data(dataset, IMG_BATCH, step);
            batch_data(labels, LABEL_BATCH, step);

            //Forward pass
            Dense1.forward(IMG_BATCH);
            Activation1.forward(Dense1.output);
            Dense2.forward(Activation1.output);
            Activation2.forward(Dense2.output);
            Dense3.forward(Activation2.output);
            softmax.forward(Dense3.output, LABEL_BATCH);

            //Backward pass
            softmax.backward(softmax.output, LABEL_BATCH);
            Dense3.backward(softmax.dinputs);
            Activation2.backward(Dense3.dinputs);
            Dense2.backward(Activation2.dinputs);
            Activation1.backward(Dense2.dinputs);
            Dense1.backward(Activation1.dinputs);

            //Update weights and biases
            optimizer.update_params(Dense1);
            optimizer.update_params(Dense2);
            optimizer.update_params(Dense3);

            //Print progress bar and record loss and accuracy
            ProgressBar(step, dataset.size() / BATCH_SIZE);
            acc[step] = accuracy(softmax.output, LABEL_BATCH);
            loss[step] = softmax.loss;
        }
        optimizer.decay_lr();

        //Print average loss and accuracy values
        cout << endl;
        cout << "Loss: " << avg(loss.data(), NUM_BATCHES) << "\tAccuracy: " << avg(acc.data(), NUM_BATCHES) << "\t"
            << "Lr: " << optimizer.current_lr;
        cout << endl;
    }

    //Calculate and print overall execution time and seconds per epoch
#ifdef _WIN32
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float delta = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
#else
    gettimeofday(&end, NULL);
    float delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
#endif

    float secsPerEpoch = delta / NUM_EPOCHS;

    printf("%.2f secs\n", delta);
    printf("%.2f secs/epoch\n", secsPerEpoch);

#ifdef _WIN32
    system("pause");
#endif

    return 0;
}
