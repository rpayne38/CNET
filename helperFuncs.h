#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <omp.h>

typedef std::vector<std::vector<double>>    Matrix2d;
typedef std::vector<double>                 Matrix1d;

double getRandomDouble(double low, double high)
{
    using namespace std;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(low, high);
    return dis(gen);
}

Matrix2d matrixMultiply(Matrix2d&x, Matrix2d &y)
{
    Matrix2d output = Matrix2d(x.size(), Matrix1d(y[0].size(), 0));
#pragma omp parallel for collapse(3)
    for (int i = 0; i < x.size(); i++)
    {
        for (int j = 0; j < y[0].size(); j++)
        {
            for (int k = 0; k < x[0].size(); k++)
            {
                output[i][j] += x[i][k] * y[k][j];
            }
        }
    }
    return output;
}

Matrix1d matrixAdd(Matrix1d &x, Matrix1d &y)
{
    Matrix1d output(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
    {
        output[i] = x[i] + y[i];
    }
    return output;
}

void printMatrix2D(Matrix2d &mat)
{
    for (unsigned int i = 0; i < mat.size(); i++)
    {
        for (unsigned int j = 0; j < mat[0].size(); j++)
        {
            std::cout << mat[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}

void printMatrix1D(Matrix1d &mat)
{
    for (unsigned int i = 0; i < mat.size(); i++)
    {
        std::cout << mat[i] << "  ";
    }
    std::cout << std::endl;
}

Matrix2d transpose(Matrix2d &mat)
{
    Matrix2d result(mat[0].size(), Matrix1d(mat.size()));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < mat.size(); i++)
    {
        for (int j = 0; j < mat[0].size(); j++)
        {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

Matrix1d sumMatrix(Matrix2d &mat, int axis)
{
    Matrix1d sum;
    if (axis == 0)
    {
        sum = Matrix1d(mat[0].size(), 0);
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
        sum = Matrix1d(mat.size(), 0);
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

Matrix1d argmax(Matrix2d &mat)
{
    Matrix1d ans(mat.size(), 0);
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
Matrix2d read_mnist_imgs(std::string path)
{
    Matrix2d dataset;
    std::ifstream file(path, std::ios::binary);
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
        dataset = Matrix2d(number_of_images, Matrix1d(n_rows * n_cols));
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

Matrix2d read_mnist_labels(std::string path)
{
    Matrix2d dataset;
    std::ifstream file(path, std::ios::binary);
    if (file.is_open())
    {
        unsigned int labels = 10;
        int magic_number = 0;
        unsigned int number_of_labels = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        dataset = Matrix2d(number_of_labels, Matrix1d(labels, 0));
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

void batch_data(Matrix2d &data, Matrix2d &batched_data, int STEP)
{
    int BATCH_SIZE = batched_data.size();
    auto start = data.begin() + (BATCH_SIZE * STEP);
    auto end = data.begin() + ((STEP + 1) * BATCH_SIZE);
    copy(start, end, batched_data.begin());
}

void ProgressBar(int step, int total)
{
    float progress = float(step + 1) / float(total);
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

double avg(double array[], int size)
{
    double avg = 0.0;
    double sum = 0.0;
    for (int i = 0; i < size; ++i)
    {
        sum += array[i];
    }
    avg = sum / size;
    return avg;
}