/*
 * Copyright(C) 2012, Thibaud Briand, ENS Cachan <thibaud.briand@ens-cachan.fr>
 * Copyright(C) 2012, Jonathan Vacher, ENS Cachan <jvacher@ens-cachan.fr>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
* @file matching_hist.c
* @brief Source code for the histogram matching.
*
* @version 0.98
* @author Thibaud BRIAND & Jonathan VACHER ;
* <thibaud.briand@ens-cachan.fr> ; <jvacher@ens-cachan.fr>
*
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
// #include "mt19937ar.h" /* Not used anymore */

/**
 * @brief Float comparison function for qsort.
 */
int f_compare (const void * a, const void * b)
{
    if(*(const float*)a < *(const float*)b)
        return -1;
    return *(const float*)a > *(const float*)b;
}

/**
* @brief Compute the sorted values of an array
*
* @param im_in input array
* @param N array length
* @return sort_values_out array of sorted values
*
* @Warning The output length is doubled since it is useful
* to keep the index of sorted values. See @ref matchHist function.
*/
void sort_values(float *sort_values_out, float *im_in, size_t N)
{
    /* mix values and indexes to keep track of pixels' location */
    for (unsigned int idx=0; idx<N; idx++) {
        sort_values_out[2*idx] = im_in[idx];
        sort_values_out[2*idx+1] = (float) 0;
    }

    /* sort pixels depending on their values*/
    qsort(sort_values_out, N , 2*sizeof(float), f_compare);
}

/**
* @brief Histogram matching of two arrays
*
* Give the histogram of an array im_ref to an other array im_out (such that the size of im_out is a multiple of the size of im_ref).
* The procedure consists in sorting the pixels of im_out and affect the to the pixel of im_out the gray-level of the pixel of im_ref that has the same rank.
*
* @param im_out modified array
* @param length_out modified array length
* @param sort_values_ref reference array of sorted values
* @param length_ref reference array length
* @return im_out with the histogram of im_ref
* @warning It is required that length_out is a multiple of length_ref !
* When the image im_out has quantized values, one can add a small noise to the image in order to induce random permutation among the pixels that share the same gray-level so that . However we do not do it here (anymore thanks to the reviewers) since im_out has float values, and thus the probability that several pixels share the same gray-level is quite low.
*/
void matchHist(float *im_out, size_t length_out,
               float *sort_values_ref, size_t length_ref)
{
    /* define the ratio: it is assumed to be an integer! */
    int ratio = length_out/length_ref;

    /* mix values and indexes to keep track of pixels' location */
    float *sort_values_out=malloc(2*length_out*sizeof(float));
    for (unsigned int idx=0; idx<length_out; idx++) {
        sort_values_out[2*idx] = im_out[idx];
        sort_values_out[2*idx+1] = (float) idx;
    }

    /* sort pixels depending on their values*/
    qsort(sort_values_out, length_out, 2*sizeof(float), f_compare);

    /* histogram matching */
    for(unsigned int idx =0; idx < length_ref ; idx++) {
        for(int k = 0; k<ratio ; k++) {
            im_out[ (int)
                    sort_values_out[2*(idx*ratio+k)+1]]=sort_values_ref[2*idx];
        }
    }

    /* free memory */
    free(sort_values_out);
}

/**
 * @brief Histogram matching of a list of arrays
 *
 * This function simplifies the code during the iteration of the algorithm.
 *
 * @param list_im_out modified list
 * @param size_out modified list size
 * @param list_sort_values_in reference list of sorted values
 * @param size_in reference list size
 * @param length_list length of the two lists (parameter used in the loop)
 * @return list_im_out with the histogram of list_im_ref (for each array)
 *
 * @note This function can be parallelized if GOMP (OpenMP) is installed.
 */
void matchHistList(float **list_im_out,size_t *size_out[2],
                   float **list_sort_values_in,
                   size_t *size_in[2], int length_list)
{
    /* loop on the whole list */
    int i;
    #pragma omp parallel for private(i) ordered schedule(dynamic)
    for(i=0; i<length_list; i++) {
        matchHist(list_im_out[i], size_out[i][0]*size_out[i][1],
                  list_sort_values_in[i], size_in[i][0]*size_in[i][1]);
    }
}
