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
* @file periodic_component.c
* @brief Source code for computing the periodic component.
*
* @version 0.98
* @author Thibaud BRIAND, Jonathan VACHER & Bruno GALERNE;
* <thibaud.briand@ens-cachan.fr> ; <jvacher@ens-cachan.fr>;
* <bruno.galerne@cmla.ens-cachan.fr>
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "eig3.h"

/* M_PI is a POSIX definition */
#ifndef M_PI
/** macro definition for Pi */
#define M_PI 3.14159265358979323846
#endif                          /* !M_PI */

/*
* number of threads to use for libfftw
* (uncomment to enable parallel FFT multi-threading)
*/
#define FFTW_NTHREADS 4 

/**
* @brief Compute the mean of a 2D array of float type.
*
* @param data_in input array
* @param N size of the array
* @return mean of data_in
*/
float mean(float *data_in, size_t N)

{
    float m = 0;
    float *ptr, *ptr_end;

    /* Pointers initialization */
    ptr = data_in;
    ptr_end = ptr + N;

    /* Sum loop */
    while(ptr < ptr_end) {
        m += *ptr;
        ptr++;
    }

    /* Normalization */
    m /= ((float) N);

    return(m);
}

/**
* @brief Pointwise multiplication of a complex array by a float array.
*
* @param comp_out output complex array
* @param comp_in input complex array
* @param float_in input float array
* @param N common size of both arrays
* @return comp_out the multiplied complex array, or NULL if an error occured
*/
fftwf_complex *pointwise_complexfloat_multiplication(
    fftwf_complex *comp_out,fftwf_complex *comp_in, float *float_in, size_t N)
{
    fftwf_complex *ptr_comp_in, *ptr_comp_end, *ptr_comp_out;
    float *ptr_float_in;

    /* check allocaton */
    if (NULL == comp_in || NULL == float_in)
        return(NULL);

    ptr_comp_in = comp_in;
    ptr_comp_out = comp_out;
    ptr_float_in = float_in;
    ptr_comp_end = ptr_comp_in + N;
    while( ptr_comp_in < ptr_comp_end ) {
        (*ptr_comp_out)[0] = (*ptr_float_in)*(*ptr_comp_in)[0];
        (*ptr_comp_out)[1] = (*ptr_float_in)*(*ptr_comp_in)[1];
        ptr_comp_in++;
        ptr_float_in++;
        ptr_comp_out++;
    }
    return(comp_out);
}

/**
* @brief Compute the discrete Laplacian of a 2D array.
*
* This function computes the discrete laplacian, ie
* @f$ (F_{i - 1, j} - F_{i + 1, j})
*     + (F_{i + 1, j} - F_{i, j})
*     + (F_{i, j - 1} - F_{i, j})
*     + (F_{i, j + 1} - F_{i, j}) \f$.
* On the border, differences with "outside of the array" are 0.
*
* This is a slightly modified version of a function written by
* Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>.
*
* @param data_out output array
* @param data_in input array
* @param nx, ny array size
* @return data_out, or NULL if an error occured
*
* @note Differences with "outside of the array" are 0.
*/
static float *discrete_laplacian(float *data_out, float *data_in,
                                 size_t nx, size_t ny)

{
    int i, j;
    float *out_ptr;
    float *in_ptr, *in_ptr_xm1, *in_ptr_xp1, *in_ptr_ym1, *in_ptr_yp1;

    /* check allocation */
    if (NULL == data_in || NULL == data_out)
        return(NULL);

    /* pointers to the data and neighbour values */
    in_ptr = data_in;
    in_ptr_xm1 = data_in - 1;
    in_ptr_xp1 = data_in + 1;
    in_ptr_ym1 = data_in - nx;
    in_ptr_yp1 = data_in + nx;
    out_ptr = data_out;
    /* iterate on j, i, following the array order */
    for (j = 0; j < (int) ny; j++) {
        for (i = 0; i < (int) nx; i++) {
            *out_ptr = 0;
            /* row differences */
            if (0 < i) {
                *out_ptr += *(in_ptr_xm1) - *in_ptr;
            }
            if ((int) nx - 1 > i) {
                *out_ptr += *(in_ptr_xp1) - *in_ptr;
            }
            /* column differences */
            if (0 < j) {
                *out_ptr += *(in_ptr_ym1) - *in_ptr;
            }
            if ((int) ny - 1 > j) {
                *out_ptr += *(in_ptr_yp1) - *in_ptr;
            }
            in_ptr++;
            in_ptr_xm1++;
            in_ptr_xp1++;
            in_ptr_ym1++;
            in_ptr_yp1++;
            out_ptr++;
        }
    }
    return(data_out);
    free(data_out);
}

/**
* @brief Compute real factor arrizing in the FFT-based discrete Poisson solver.
*
* We have :
* \f[
out(\xi_1,\xi_2) =\frac{1}{N}\frac{1}{2\cos\left( \frac{2 \xi_1 \pi}{M} \right)
+ 2\cos\left( \frac{2 \xi_2 \pi}{N} \right) - 4}
\f]
* if
* @f$ (\xi_1,\xi_2) \neq (0,0) \f$
* and
* @f$ out(0,0) = 1 \f$.
* Each coefficient is the inverse of an eigenvalue of the periodic Laplacian.
*
* The factor @f$ 1/N \f$ is added to normalize the Fourier transform
* before the inversion (note that FFTW does NOT normalize the transforms).
*
* @param data_out output float array
* @param nx, ny input size
* @return data_out, or NULL if an error occured
* @warning The ouput array data_out has size (nx/2+1)*ny to be
* compatible with the R2C FFT (which takes advantage of
* the symmetry of the DFT of real data).
*/
static float *poisson_complex_filter(float *data_out, size_t nx, size_t ny)

{
    float *cosx, *cosymintwo, *ptr_x, *ptr_y, *ptr_x_end, *ptr_y_end, *out_ptr;
    int i, j;
    size_t sx = nx/2 + 1;
    int N = nx*ny;
    float halfinvN = 0.5/((float) N);

    /* check allocation */
    if (NULL == data_out)
        return(NULL);

    /* allocate memory */
    if (NULL == (cosx = (float *) malloc(sx * sizeof(float)))
            ||NULL == (cosymintwo = (float *) malloc(ny * sizeof(float))))
        return(NULL);

    /* define cosx and cosymintwo */
    ptr_x = cosx;
    for(i=0; i< (int) sx; i++) {
        (*ptr_x) = cos(2. * ((float) i) * M_PI /((float) nx));
        ptr_x++;
    }
    ptr_y = cosymintwo;
    for(j=0; j< (int) ny; j++) {
        (*ptr_y) = cos(2.* ((float) j) * M_PI/((float) ny)) - 2.;
        ptr_y++;
    }

    /* define data_out */
    out_ptr = data_out;
    ptr_y = cosymintwo;
    ptr_y_end = cosymintwo + ny;
    ptr_x = cosx;
    ptr_x_end = cosx + sx;
    /* particular case : (0,0) */
    (*out_ptr) = 1.;
    out_ptr++;
    ptr_x++;
    while(ptr_y < ptr_y_end) {
        while(ptr_x < ptr_x_end) {
            (*out_ptr) = halfinvN / ( (*ptr_x) + (*ptr_y) );
            out_ptr++;
            ptr_x++;
        }
        ptr_x = cosx;
        ptr_y++;
    }

    /* free memory */
    free(cosx);
    free(cosymintwo);

    /* return data_out */
    return(data_out);
    free(data_out);
}

/**
* @brief Compute the periodic component of an image.
*
* This function computes Lionel Moisan's
* periodic component of an image in solving a
* Poisson equation with forward and backward FFT.
*
* \b Reference: L. Moisan, "Periodic plus Smooth Image Decomposition",
* preprint MAP5, Université Paris Descartes, 2009,
* available at <http://www.math-info.univ-paris5.fr/~moisan/p+s/>.
*
* Computation :
* @li a discrete Laplacian is computed;
* @li the discrete Laplacian is transformed by forward DFT;
* @li the obtained DFT is modified by
* @f[ \hat{u}(i, j) = \hat{F}(i, j) / (4 - 2 \cos(\frac{2 i \pi}{n_x})
* - 2 \cos(\frac{2 j \pi}{n_y})), @f]
* and
* @f$ \hat{u}(0, 0) @f$ is set to be the sum of the input;
* @li this data is transformed by backward DFT.
*
* Below is an illustration of a color image and its periodic component.
*
* @image html perdecomp.png "Input image (left) and
* its periodic component (right)"
*
* @param data_in input float array.
* @param nx, ny array size
* @return data_in_per or NULL if an error occured
*/

float *periodic_component(float *data_in_per, float *data_in, size_t nx, size_t ny)
{
    float *laplacian; /* 2D discrete Laplacian */
    float *pcf; /* Poisson complex filter */
    fftwf_plan plan_r2c, plan_c2r;
    fftwf_complex *fft, *fft2;
    int N = ((int) nx)*((int) ny);
    float m; /* mean of the coefficient of data_in */
    size_t fft_size = (nx/2+1)*ny; /* physical size of the r2c FFT */

    /* Check allocation */
    if( NULL == data_in )
        return(NULL);

    /* Start threaded fftw if FFTW_NTHREADS is defined */
#ifdef FFTW_NTHREADS
    if( 0 == fftwf_init_threads() )
        return(NULL);
    fftwf_plan_with_nthreads(FFTW_NTHREADS);
#endif

    /* Allocate memory */
    if( NULL == (pcf = (float *) malloc(fft_size * sizeof(float)))
            || NULL == (laplacian = (float *) malloc(N * sizeof(float)))
            || NULL == (fft = (fftwf_complex *)
                              fftwf_malloc( fft_size * sizeof(fftwf_complex)))
            || NULL == (fft2 = (fftwf_complex *)
                               fftwf_malloc(fft_size * sizeof(fftwf_complex))))
        return(NULL);

    /* Compute the Poisson complex filter */
    if( NULL == poisson_complex_filter(pcf, nx, ny) )
        return(NULL);


    /* Compute m = the mean value of data_in*/
    m = mean(data_in, N);

    /* Compute the discrete Laplacian of data_in */
    if(NULL == discrete_laplacian(laplacian, data_in, nx, ny))
        return(NULL);

    /* Forward Fourier transform of the Laplacian: create the FFT forward
     * plan and run the FFT */
    plan_r2c = fftwf_plan_dft_r2c_2d((int) (ny), (int) (nx),
                                     laplacian, fft,
                                     FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    fftwf_execute(plan_r2c);

    /* Inverse the discrete periodic Laplacian in the Fourier domain using
     *the Poisson complex filter */
    if( NULL == pointwise_complexfloat_multiplication(fft2,fft, pcf, fft_size))
        return(NULL);
    /* (0,0) frequency : we impose the same mean as the input */
    (*fft2)[0] = m;
    (*fft2)[1] = 0;

    /* Backward Fourier transform: the output is stored
     * in data_in_rgb[channel] */
    plan_c2r = fftwf_plan_dft_c2r_2d((int) (ny), (int) (nx),
                                     fft2, data_in_per,
                                     FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    fftwf_execute(plan_c2r);

    /* cleanup */
    free(pcf);
    free(laplacian);
    /* destroy the FFT plans and data */
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
    fftwf_free(fft);
    fftwf_free(fft2);
    fftwf_cleanup();
#ifdef FFTW_NTHREADS
    fftwf_cleanup_threads();
#endif
    return(data_in_per);
}
