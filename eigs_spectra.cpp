#include <Eigen/Core>
#include <Spectra/SymEigsShiftSolver.h>
// <Spectra/MatOp/DenseSymShiftSolve.h> is implicitly included
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "eigs.h"

// Inspired by https://github.com/yixuan/spectra/blob/master/include/Spectra/SymEigsShiftSolver.h

#define MIN(x, y) (x < y ? x : y)

using namespace Spectra;

void smallest_eigenvalues(double *A, int n, int k, double *ret_eigenvalues, double *ret_eigenvectors) {
    Eigen::MatrixXd M = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(A, n, n);
    // Construct matrix operation object using the wrapper class
    DenseSymShiftSolve<double> op(M);
    // Construct eigen solver object with shift 0
    // This will find eigenvalues that are closest to 0
    SymEigsShiftSolver< double, LARGEST_MAGN,
                        DenseSymShiftSolve<double> > eigs(&op, k, MIN(2*k, n), 0.0);
    eigs.init();
    eigs.compute();
    if(eigs.info() == SUCCESSFUL)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues();
        Eigen::MatrixXd evectors = eigs.eigenvectors().transpose();
        if (ret_eigenvalues) memcpy(ret_eigenvalues, evalues.data(), k * sizeof(double));
        if (ret_eigenvectors) memcpy(ret_eigenvectors, evectors.data(), n * k * sizeof(double));
        // std::cout << "Eigenvalues found:\n" << evalues << std::endl;
        // std::cout  << "\nEigenvectors found:\n" << evectors << std::endl;
        return;
    } 
    fprintf(stderr, "\033[31mWARNING: Spectra failed to compute eigenvalues\033[0m\n");
}

// int main(int argc, char *argv[]) {
//     int n = 1000;
//     int k = 10;
//     double *A = (double *) malloc(n * n * sizeof(double));
//     for (int i = 0; i < n; i++) {
//         A[i * n + i] = i+1;
//     }
//     double *eigenvectors = (double *) malloc(n * k * sizeof(double));
//     smallest_eigenvalues_double(A, n, k, NULL, eigenvectors);
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < k; j++) {
//             printf("%1.5f ", eigenvectors[i * k + j]);
//         }
//         printf("\n");
//     }
// }