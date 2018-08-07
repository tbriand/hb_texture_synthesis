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
* @file filters.c
* @brief Compute the filters used during the analysis & synthesis steps.
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
 * @brief Compute the constant of reversibility alpha.
 *
 * We use the product rule of log to compute this constant.
 *
 * @param N_steer number of orientation bands
 * @return alpha_K the constant of reversibility
 */
float compute_alpha(int N_steer)
{
    /* allocation */
    float log_fact_alpha1=0;
    float log_fact_alpha2 =0;
    float log_alpha;
    float alpha_K;

    /* compute the two sums */
    for(int k=2; k<N_steer; k++) log_fact_alpha1 += log(k);
    for(int k=2; k<2*N_steer-1; k++) log_fact_alpha2 += log(k);

    /* compute the log of alpha */
    log_alpha = (N_steer-1)*log(2)+ log_fact_alpha1
                - 0.5*(log(N_steer)+log_fact_alpha2);

    /* compute alpha */
    alpha_K = exp(log_alpha);

    return(alpha_K);
}

/**
* @brief Compute the high pass filter (in the Fourier domain).
*
* @image html high.png "High pass filter in the Fourier domain"
*
* @param nx,ny filter sizes
* @param factor is used to define the first high pass filter or the loop one
* @return data_out_high_pass the high pass filter (float array)
*/
static void high_filters(float *data_out_high_pass, size_t nx, size_t ny, float factor)
{
    float x,y,r;

    /* polar definition of the filter */
    for(unsigned int j=0; j<nx; j++) {
        for(unsigned int k=0; k<ny; k++) {

            x=(j<nx/2) ? j*2*M_PI/(nx):j*2*M_PI/(nx)-2*M_PI;
            y=(k<ny/2) ? k*2*M_PI/(ny):k*2*M_PI/(ny)-2*M_PI;
            r=hypot(x,y)/factor;

            if(r!=0) {
                data_out_high_pass[j+k*nx] = ((r>=M_PI/2) +
                                              (r>M_PI/4)*(r<M_PI/2)*cos(M_PI/2*
                                                      (log(2*r/M_PI)/log(2))));
            } else {
                data_out_high_pass[j+k*nx]=0;
            }
        }
    }
}

/**
* @brief Compute the low pass filter (in the Fourier domain).
*
* @image html low.png "Low pass filter in the Fourier domain"
*
* @param nx,ny filter size
* @param factor is used to define the first low pass filter or the loop one
* @return data_out_low_pass the high pass filter (float array)
*/
static void low_filters(float *data_out_low_pass, size_t nx, size_t ny, float factor)
{
    float x,y,r;

    /* polar definition of the filter */
    for(unsigned int j=0; j<nx; j++) {
        for(unsigned int k=0; k<ny; k++) {

            x=(j<nx/2) ? j*2*M_PI/(nx):j*2*M_PI/(nx)-2*M_PI;
            y=(k<ny/2) ? k*2*M_PI/(ny):k*2*M_PI/(ny)-2*M_PI;
            r=hypot(x,y)/factor;

            if(r!=0) {
                data_out_low_pass[j+k*nx] = ((r<=M_PI/4) +
                                             (r>M_PI/4)*(r<M_PI/2)*cos(M_PI/2*
                                                     (log(4*r/M_PI)/log(2))));
            } else {
                data_out_low_pass[j+k*nx]=1;
            }
        }
    }
}

/**
* @brief Compute the steered filters (in the Fourier domain).
*
* @image html filtro2.png "Steered filter in the Fourier domain"
*
* @param nx,ny filter size
* @param steer the orientation of the filter
* @param N_steer number of orientation bands
* @param alpha constant of reversibility
* @return data_out_steer_pass the steered filter with orientation 'steer'
*/

static void steerable_filters(float *data_out_steer_pass, size_t nx, size_t ny,
                              int steer, int N_steer, float alpha)
{
    float x,y,theta,factor,cosinus_theta,sign;

    /* polar definition of the filter */
    for(unsigned int j=0; j<nx; j++) {
        for(unsigned int k=0; k<ny; k++) {
            if(j==0 && k==0) data_out_steer_pass[0] = 1;
            else {
                x=(j<nx/2) ? j*2*M_PI/(nx):j*2*M_PI/(nx)-2*M_PI;
                y=(k<ny/2) ? k*2*M_PI/(ny):k*2*M_PI/(ny)-2*M_PI;
                theta=atan2(y,x);
                factor=cos(theta-M_PI*steer/N_steer);
                cosinus_theta=1;
                sign=1;

                /* computation of a power */
                for( int p = 1 ; p<N_steer ; p++) {
                    cosinus_theta *= factor;
                    sign *= -1;
                }

                data_out_steer_pass[j+k*nx] = (float)
                                              (alpha*cosinus_theta*
                                               (fabs(fmod(theta+3*M_PI-
                                                       M_PI*steer/N_steer,2*M_PI)
                                                     -M_PI)
                                                <M_PI/2)
                                               +alpha*sign*cosinus_theta*
                                               (fabs(fmod(theta+2*M_PI
                                                       -M_PI*steer/N_steer,2*M_PI)
                                                     -M_PI)
                                                <M_PI/2));
            }
        }
    }
}

/**
* @brief Compute a list which contains the sizes of the filters.
*
* @param nx,ny sample size
* @param N_steer number of orientation bands
* @param N_pyr number of scales
* @return size_list, size_list_downsample the size of the filters
* used during the analysis and the downsampling
*/

void size_filters(size_t **size_list,
                  size_t **size_list_downsample,size_t nx,size_t ny,
                  int N_steer, int N_pyr)
{
    /* define the size of the first filters */
    size_list[0][0]= nx;
    size_list[0][1]= ny;
    size_list_downsample[0][0]=nx;
    size_list_downsample[0][1]=ny;

    /* define the size of the filters used during the loop*/
    for(int k=0; k<N_pyr; k++) {
        size_list_downsample[k+1][0]= (size_t)  nx/(1<<k);
        size_list_downsample[k+1][1]= (size_t)  ny/(1<<k);

        for(int j=0; j<N_steer; j++) {
            size_list[1+N_steer*k+j][0]= (size_t)  nx/(1<<k);
            size_list[1+N_steer*k+j][1]= (size_t)  ny/(1<<k);
        }
    }

    /* define the size of the last filter */
    size_list[1+N_steer*N_pyr][0]= (size_t) nx/(1<<(N_pyr));
    size_list[1+N_steer*N_pyr][1]= (size_t) ny/(1<<(N_pyr));
}

/**
* @brief Compute the filters of the steerable pyramid.
*
* @param nx,ny sample size
* @param N_steer number of orientation bands
* @param N_pyr number of scales
* @param size_list_downsample list of filters sizes (downsampling)
* @param size_list list of filters sizes (analysis)
* @param alpha constant of reversibility
* @return list_filters, list_downsample_filters the filters
* used during the analysis and the downsampling
*/
void filters(float **list_filters, float **list_downsample_filters,
             size_t *size_list[2], size_t *size_list_downsample[2],
             size_t nx,size_t ny ,int N_steer, int N_pyr, float alpha)
{
    /* memory allocation */
    float *high_filters_loop = malloc(nx*ny*sizeof(float));
    float *steer_filters_loop = malloc(nx*ny*sizeof(float));

    /* Fill the list of filters */
    /* Compute the first and last filters */
    high_filters(list_filters[0],nx,ny,2);
    low_filters(list_filters[1+N_steer*N_pyr],size_list[1+N_steer*N_pyr][0],
                size_list[1+N_steer*N_pyr][1], 1);
    low_filters(list_downsample_filters[0],nx,ny,2);

    /* compute the others filters : loop on the scales */
    for(int k=0; k<N_pyr; k++) {
        low_filters(list_downsample_filters[1+k],size_list_downsample[1+k][0],
                    size_list_downsample[1+k][1],1);

        /* high filters used during the loop */
        high_filters(high_filters_loop,size_list[1+k*N_steer][0],
                     size_list[1+k*N_steer][1],1);

        /* loop on the steers */
        for(int j=0; j<N_steer; j++) {
            steerable_filters( steer_filters_loop, size_list[1+k*N_steer][0],
                               size_list[1+k*N_steer][1],
                               j,N_steer, alpha);

            for(unsigned int l=0;
                    l<size_list[1+k*N_steer][0]*size_list[1+k*N_steer][1]; l++)
                list_filters[1+k*N_steer+j][l] = steer_filters_loop[l]*
                                                 high_filters_loop[l];
        }
    }

    /* free */
    free(high_filters_loop);
    free(steer_filters_loop);
}
