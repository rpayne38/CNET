#include "CNet.h"
#include <sys/time.h>
using namespace std;

int epochs = 5;
int BATCH_SIZE = 1000;

int main()
{
     //Start timer
     struct timeval start, end;
     gettimeofday(&start, NULL);

     cout << "Loading data...\n";
     vector<vector<double>> dataset = read_mnist_imgs("train-images.idx3-ubyte");
     vector<vector<double>> labels = read_mnist_labels("train-labels.idx1-ubyte");
     int NUM_BATCHES = dataset.size() / BATCH_SIZE;

     cout << "Loading model...\n";
     Dense Dense1(28 * 28, 32);
     Relu Activation1;
     Dense Dense2(32, 32);
     Relu Activation2;
     Dense Dense3(32, 10);
     SoftmaxwithLoss softmax;
     SGD optimizer(0.01, 0.05, 0.9);

     for (int epoch = 0; epoch < epochs; epoch++)
     {
          double acc[NUM_BATCHES]{};
          double loss[NUM_BATCHES]{};
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
          cout << "Loss: " << avg(loss, NUM_BATCHES) << "\tAccuracy: " << avg(acc, NUM_BATCHES) << "\t"
               << "Lr: " << optimizer.current_lr;
          cout << endl;
     }

     //Calculate and print overall execution time and seconds per epoch
     gettimeofday(&end, NULL);
     float delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
     cout << delta << "secs"
          << "\n";
     cout << delta / epochs << " secs/epoch"
          << "\n";
     return 0;
}
