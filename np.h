#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
using namespace std;

double getRandomDouble(double low, double high)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(low, high);
    return dis(gen);
}

vector<vector<double>> matrixMultiply(vector<vector<double>> &x, vector<vector<double>> &y)
{
    vector<vector<double>> output = vector<vector<double>>(x.size(), vector<double>(y[0].size(), 0));
#pragma omp parallel for collapse(3)
    for (unsigned int i = 0; i < x.size(); i++)
    {
        for (unsigned int j = 0; j < y[0].size(); j++)
        {
            for (unsigned int k = 0; k < x[0].size(); k++)
            {
                output[i][j] += x[i][k] * y[k][j];
            }
        }
    }
    return output;
}

vector<double> matrixAdd(vector<double> &x, vector<double> &y)
{
    vector<double> output(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
    {
        output[i] = x[i] + y[i];
    }
    return output;
}

void printMatrix2D(vector<vector<double>> &mat)
{
    for (unsigned int i = 0; i < mat.size(); i++)
    {
        for (unsigned int j = 0; j < mat[0].size(); j++)
        {
            cout << mat[i][j] << "  ";
        }
        cout << endl;
    }
}

void printMatrix1D(vector<double> &mat)
{
    for (unsigned int i = 0; i < mat.size(); i++)
    {
        cout << mat[i] << "  ";
    }
    cout << endl;
}

vector<vector<double>> transpose(vector<vector<double>> &mat)
{
    vector<vector<double>> result(mat[0].size(), vector<double>(mat.size()));
    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < mat.size(); i++)
    {
        for (unsigned int j = 0; j < mat[0].size(); j++)
        {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

vector<double> sumMatrix(vector<vector<double>> &mat, int axis)
{
    vector<double> sum;
    if (axis == 0)
    {
        sum = vector<double>(mat[0].size(), 0);
        for (unsigned int row = 0; row < mat.size(); row++)
        {
            for (unsigned int col = 0; col < mat[0].size(); col++)
            {
                sum[col] += mat[row][col];
            }
        }
    }
    else if (axis == 1)
    {
        sum = vector<double>(mat.size(), 0);
        for (unsigned int row = 0; row < mat.size(); row++)
        {
            for (unsigned int col = 0; col < mat[0].size(); col++)
            {
                sum[row] += mat[row][col];
            }
        }
    }
    return sum;
}

vector<double> argmax(vector<vector<double>> &mat)
{
    vector<double> ans(mat.size(), 0);
    for (unsigned int row = 0; row < mat.size(); row++)
    {
        double max = 0;
        for (unsigned int col = 0; col < mat[0].size(); col++)
        {
            if (mat[row][col] > max)
            {
                max = mat[row][col];
                ans[row] = col;
            }
        }
    }
    return ans;
}

int reverseInt(int i)
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
    ifstream file(path, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        unsigned int number_of_images = 0;
        unsigned int n_rows = 0;
        unsigned int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        dataset = vector<vector<double>>(number_of_images, vector<double>(n_rows * n_cols));
        for (unsigned int i = 0; i < number_of_images; ++i)
        {
            for (unsigned int r = 0; r < n_rows; ++r)
            {
                for (unsigned int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char *)&temp, sizeof(temp));
                    dataset[i][(28 * r) + c] = temp;
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
    ifstream file(path, ios::binary);
    if (file.is_open())
    {
        unsigned int labels = 10;
        int magic_number = 0;
        unsigned int number_of_labels = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        dataset = vector<vector<double>>(number_of_labels, vector<double>(labels, 0));
        for (unsigned int i = 0; i < number_of_labels; ++i)
        {
            {
                unsigned char temp = 0;
                file.read((char *)&temp, sizeof(temp));
                dataset[i][temp] = 1;
            }
        }
    }
    file.close();
    return dataset;
}