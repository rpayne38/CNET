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

vector<double> matrixAdd(vector<double> &x, vector<double> &y)
{
    vector<double> output(x.size());
    #pragma omp parallel for 
    for (int i = 0; i < x.size(); i++)
    {
        output[i] = x[i] + y[i];
    }
    return output;
}

void printMatrix2D(vector<vector<double>> &mat)
{
    for (int i = 0; i < mat.size(); i++)
    {
        for (int j = 0; j < mat[0].size(); j++)
        {
            cout << mat[i][j] << "  ";
        }
        cout << endl;
    }
}

void printMatrix1D(vector<double> &mat)
{
    for (int i = 0; i < mat.size(); i++)
    {
        cout << mat[i] << "  ";
    }
    cout << endl;
}

vector<vector<double>> transpose(vector<vector<double>> &mat)
{
    vector<vector<double>> result(mat[0].size(), vector<double>(mat.size()));
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

vector<double> sumMatrix(vector<vector<double>> &mat, int axis)
{
    vector<double> sum;
    if (axis == 0)
    {
        vector<double> sum = vector<double>(mat[0].size(), 0);
        #pragma omp parallel for collapse(2) 
        for (int row = 0; row < mat.size(); row++)
        {
            for (int col = 0; col < mat[0].size(); col++)
            {
                sum[col] += mat[row][col];
            }
        }
        return sum;
    }
    else if (axis == 1)
    {
        vector<double> sum = vector<double>(mat.size(), 0);
        #pragma omp parallel for collapse(2) 
        for (int row = 0; row < mat.size(); row++)
        {
            for (int col = 0; col < mat[0].size(); col++)
            {
                sum[row] += mat[row][col];
            }
        }
        return sum;
    }
    else
    {
        cout << "Something Broke";
    }
}

vector<double> argmax(vector<vector<double>> &mat)
{
    vector<double> ans(mat.size(), 0);
    for (int row = 0; row < mat.size(); row++)
    {
        double max = 0;
        for (int col = 0; col < mat[0].size(); col++)
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