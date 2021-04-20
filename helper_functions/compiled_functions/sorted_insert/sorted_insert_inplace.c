/* --------------------------------------------------------------------

     File: sorted_insert_inplace.c

     sorted_insert_inplace inserts vector elements into a sorted matrix by
     columns to result in a sorted matrix by columns. The operation is done
     inplace on the matrix A, and what would have been the last row after the
     insertion is discarded. A vector is returned that contains the indexes
     of the insertion points. Note that if an insertion point is at the end
     of a column, the insertion will not happen (since it would have been in
     the last row) and the index returned for that spot will be size(A,1)+1.
     Syntax:

     C = sorted_insert_inplace(A,B)

     A = full real double 2D matrix that is sorted by columns (MxN)
     B = full real double 1D vector that has same number of elements
         as A has columns (1xN) or (Nx1)
     C = Vector that contains the indexes of the insertions (1xN)

     The A result will be the equivalent of:
       A = sort([A;B],1);
       A(end,:) = [];

     CAUTION: If the A matrix is sharing data with another variable, then
     that other variable will be changed inplace also. It is up to the user
     to ensure that there are no other variables sharing data with matrix A,
     since this routine does not check for this condition.

     Programmer: James Tursa
     Date:  March 12, 2018
  */

/* Includes ----------------------------------------------------------- */

#include "mex.h"

/* Prototype in case this is earlier than R2015a */

mxArray *mxCreateUninitNumericMatrix(size_t m, size_t n, 
  mxClassID classid, mxComplexity ComplexFlag);

/* Gateway ------------------------------------------------------------ */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize i, i1, i2, i3, j, m, n;
    double *Apr, *Bpr, *Cpr;
    double temp1, temp2;

      if( nlhs > 1 ) {
          mexErrMsgTxt("Too many outputs");
      }
      if( nrhs != 2 || !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) ||
          mxIsComplex(prhs[0]) || mxIsSparse(prhs[0]) ||
          mxIsComplex(prhs[1]) || mxIsSparse(prhs[1]) ) {
          mexErrMsgTxt("Need two full real double inputs");
      }
      if( mxGetNumberOfDimensions(prhs[0]) != 2 ) {
          mexErrMsgTxt("First input must be a 2D matrix");
      }
      m = mxGetM(prhs[0]);
      n = mxGetN(prhs[0]);
      if( (mxGetM(prhs[1]) != 1 && mxGetN(prhs[1]) != 1) ||
          mxGetNumberOfElements(prhs[1]) != n ) {
          mexErrMsgTxt("Second input must be vector with same number of elements as columns of first input");
      }
      plhs[0] = mxCreateUninitNumericMatrix(1,n,mxDOUBLE_CLASS,mxREAL);
      Apr = mxGetPr(prhs[0]);
      Bpr = mxGetPr(prhs[1]);
      Cpr = mxGetPr(plhs[0]);
      for( j=0; j<n; j++ ) { /* For each column */
          i1 = 0;
          i3 = m;
          while( i3-i1 > 1 ) { /* Binary search to find insert spot */
              i2 = (i3 + i1) >> 1;
              if( Apr[i2] > *Bpr ) {
                  i3 = i2;
              } else {
                  i1 = i2;
              }
          }
          if( i1 < m && Apr[i1] > *Bpr ) { /* Potential 1st spot adjustment */
              i3--;
          }
          *Cpr++ = i3+1;
          Apr += i3;
          if( i3 < m ) {
              temp1 = *Apr;
              *Apr++ = *Bpr++;
              for( i=i3+1; i<m; i++ ) { /* Copy the stuff after */
                  temp2 = *Apr;
                  *Apr++ = temp1;
                  temp1 = temp2;
              }
          } else {
              Bpr++;
          }
      }
  }