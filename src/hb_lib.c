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
* @file hb_lib.c
* @brief Source code for Heeger & Bergen texture synthesis.
*
* @version 1.00
* @author Thibaud BRIAND & Jonathan VACHER ;
* <thibaud.briand@ens-cachan.fr> ; <jvacher@ens-cachan.fr>
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "eig3.h"
#include "filters.h"
#include "periodic_component.h"
#include "matching_hist.h"
#include "bilinear_zoom.h"

/* M_PI is a POSIX definition */
#ifndef M_PI
/** macro definition for Pi */
#define M_PI 3.14159265358979323846
#endif                          /* !M_PI */

/*
* number of threads to use for libfftw
* (uncomment to enable parallel FFT multi-threading)
*/
#define FFTW_NTHREADS  4

/**
* @brief Compute the DFT of a 2D real array.
* @param data_in input array
* @param nx,ny array size
* @return fft_out complex array which corresponds to the DFT of the input
*/
static void do_fft(fftwf_complex *fft_out,float *data_in, size_t nx, size_t ny)
{
    /* memory allocation */
    fftwf_plan plan_r2c;
    fftwf_complex *complex_data_in = fftwf_malloc(
                                         nx*ny * sizeof(fftwf_complex));

    /* Real --> complex */
    for(unsigned int i=0; i<nx*ny; i++) {
        complex_data_in[i][0]=data_in[i];
        complex_data_in[i][1]=0;
    }

    /* compute the DFT */
    plan_r2c = fftwf_plan_dft_2d((int) ny ,(int) nx ,
                                 complex_data_in, fft_out ,
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan_r2c);
    
    /* normilization */
    float invN = 1/((float) nx*ny);
    for(unsigned int i=0; i <nx*ny; i++){
        fft_out[i][0] = invN*fft_out[i][0];
        fft_out[i][1] = invN*fft_out[i][1];
    }

    /* free */
    fftwf_destroy_plan(plan_r2c);
    fftwf_free(complex_data_in);
    fftwf_cleanup();
}


/**
* @brief Compute the iDFT of a 2D symmetric complex array.
* @param fft_in input array
* @param nx,ny array size
* @return data_out real array which corresponds to the iDFT of the input
*/
static void do_ifft(fftwf_complex *fft_in,float *data_out, size_t nx,size_t ny)
{
    /* memory allocation */
    fftwf_plan plan_c2r;
    fftwf_complex *complex_data_out = fftwf_malloc (
                                          sizeof ( fftwf_complex ) * nx * ny );

    /* compute the iDFT */
    plan_c2r = fftwf_plan_dft_2d ( (int) ny, (int) nx , fft_in ,
                                   complex_data_out ,
                                   FFTW_BACKWARD, FFTW_ESTIMATE );
    fftwf_execute(plan_c2r);

    /* complex --> real */
    for(unsigned int i=0; i<nx*ny; i++) {
        data_out[i]=complex_data_out[i][0];
    }

    /* free */
    fftwf_free(complex_data_out);
    fftwf_destroy_plan(plan_c2r);
    fftwf_cleanup();
}

/**
* @brief Extend an fft array by zero-padding by a factor 2 (corresponds to upsampling by a factor 2 in the spatial domain)
* @param fft_in fftwf complex array of size nx x ny
* @param nx,ny size of input complex array
* @return fft_out complex array of size 2*nx x 2*ny
*/
static void upsample_fft(fftwf_complex *fft_out, fftwf_complex *fft_in, size_t nx, size_t ny)
{
    unsigned int kk, ll; /* new coordinates for original pixel (k,l) */
    unsigned int mx, my;
    mx = (unsigned int) ceil(((double) nx)/2); /* mx = p if nx = 2*p and p+1 if nx = 2*p+1 */
    my = (unsigned int) ceil(((double) ny)/2);
    
    /* Fill the output array with zeros */
    for(unsigned int i=0; i<4*nx*ny; i++){
        fft_out[i][0] = 0.;
        fft_out[i][1] = 0.;
    }
    
    for(unsigned int l=0; l<ny; l++){
        ll = (l<my) ? l : l + ny;
        for(unsigned int k=0; k<nx; k++){
            kk = (k<mx) ? k : k + nx;
            fft_out[ll*2*nx+kk][0] = fft_in[l*nx+k][0];
            fft_out[ll*2*nx+kk][1] = fft_in[l*nx+k][1];
        }
    }
}


/** 
 * @brief Extract the center of an fft array with a factor 2 (corresponds to downsampling by a factor 2 in the spatial domain)
 * @param fft_in fftwf complex array of size nx x ny
 * @param nx,ny size of input complex array
 * @return fft_out complex array of size nx/2 x ny/2
 */
static void downsample_fft(fftwf_complex *fft_out, fftwf_complex *fft_in, size_t nx, size_t ny)
{
    unsigned int kk, ll; /* original coordinates corresponding to new pixel (k,l) */
    unsigned int mx = nx/2;
    unsigned int my = ny/2;
    unsigned int mmx, mmy; /* middle on grid mx x my */
    mmx = (unsigned int) ceil(((double) mx)/2); /* mx = p if nx = 2*p and p+1 if nx = 2*p+1 */
    mmy = (unsigned int) ceil(((double) my)/2);
    
    /* check if size integers are even */
    if((nx % 2)||(ny % 2)) {printf("WARNING: Integer nx and y are supposed to be even\n"); abort();}
    
    /* fill fft_out array with the center of fft_in */
    for(unsigned int l=0; l<my; l++){
        ll = (l<mmy) ? l : l + my;
        for(unsigned int k=0; k<mx; k++){
            kk = (k<mmx) ? k : k + mx;
            fft_out[l*mx+k][0] = fft_in[ll*nx+kk][0];
            fft_out[l*mx+k][1] = fft_in[ll*nx+kk][1];
        }
    }
    
}

/**
* @brief Make the analysis of a 2D array.
*
* The output is a list of 2+N_pyr*N_steer arrays.
* The computation takes place in the Fourier domain.
*
* @image html analysis.png Example of analysis (2 scales, 4 steers)
*
* @param data_in input array (that you want to analyse)
* @param list_filters, list_downsample_filters filters used during the
* analysis and the downsampling
* @param size_list, size_list_downsample list of filters sizes used
* during the analysis and the downsampling
* @param nx,ny array size
* @param N_steer number of orientation bands
* @param N_pyr number of scales
* @return list_out the list of oriented bandpass images
*/
static void analysisHb(float **list_out,
                       float *data_in,
                       float **list_filters,
                       float **list_downsample_filters,
                       size_t *size_list[2],
                       size_t nx,size_t ny,
                       int N_steer, int N_pyr)
{
    /* allocate memory */
    size_t sx=0;
    size_t sy=0;
    fftwf_complex *fft_data_in = fftwf_malloc( nx*ny * sizeof(fftwf_complex));
    fftwf_complex *fft_data_in_high = fftwf_malloc(
                                          nx*ny * sizeof(fftwf_complex));
    fftwf_complex *fft_data_in_loop = fftwf_malloc(
                                          nx*ny * sizeof(fftwf_complex));
    fftwf_complex *fft_data_in_loop_loop = fftwf_malloc(
            nx*ny* sizeof(fftwf_complex));
    fftwf_complex *fft_data_in_loop_downsampled =
        fftwf_malloc(nx*ny*sizeof( fftwf_complex));

    /* Start threaded fftw if FFTW_NTHREADS is defined */
#ifdef FFTW_NTHREADS
    if( 0 == fftwf_init_threads() )
        printf("Problem with the multi-threads initialization of fftw");
    fftwf_plan_with_nthreads(FFTW_NTHREADS);
#endif

    /* Compute the dft of the input */
    do_fft(fft_data_in, data_in, nx, ny);

    /* high component */
    pointwise_complexfloat_multiplication(fft_data_in_high,
                                          fft_data_in,
                                          list_filters[0],
                                          nx*ny);
    do_ifft(fft_data_in_high, list_out[0], nx, ny);

    /* first low component */
    pointwise_complexfloat_multiplication(fft_data_in_loop,
                                          fft_data_in,
                                          list_downsample_filters[0],
                                          nx*ny);

    /* recursive loop to compute the oriented bandpass images*/
    for(int i=0; i<N_pyr; i++) {
        for(int j=0; j<N_steer; j++) {
            /* image size */
            sx=size_list[1+i*N_steer+j][0];
            sy=size_list[1+i*N_steer+j][1];

            /* apply the filters */
            pointwise_complexfloat_multiplication(fft_data_in_loop_loop,
                                                  fft_data_in_loop,
                                                  list_filters[1+i*N_steer+j],
                                                  sx*sy);
            /* iDFT */
            do_ifft(fft_data_in_loop_loop, list_out[1+i*N_steer+j], sx, sy);
        }

        /* DFT for the next level */
        pointwise_complexfloat_multiplication(fft_data_in_loop,
                                              fft_data_in_loop,
                                              list_downsample_filters[1+i],
                                              sx*sy);

        /* downsample */
        downsample_fft(fft_data_in_loop_downsampled, fft_data_in_loop, sx, sy);
        /* copy on fft_data_in_loop */
        for(unsigned int k=0; k<sx*sy/4; k++) {
                fft_data_in_loop[k][0] = fft_data_in_loop_downsampled[k][0];
                fft_data_in_loop[k][1] = fft_data_in_loop_downsampled[k][1];
        }

    }
    /* Compute the last low component */
    do_ifft(fft_data_in_loop, list_out[1+N_pyr*N_steer], size_list[1+N_pyr*N_steer][0], size_list[1+N_pyr*N_steer][1]);

    /* free memory */
    fftwf_free(fft_data_in_loop);
    fftwf_free(fft_data_in_high);
    fftwf_free(fft_data_in);
    fftwf_free(fft_data_in_loop_loop);
    fftwf_free(fft_data_in_loop_downsampled);
}

/**
* @brief Make the synthesis of a list of oriented bandpass images.
*
* The output is a 2D real array.
* The computation takes place in the Fourier domain.
*
* @param list_in input array (that you want to analyse)
* @param list_filters, list_upsample_filters filters used during
* the synthesis and the upsampling
* @param size_list, size_list_upsample list of filters sizes used
* during the synthesis and the upsampling
* @param nx,ny array size
* @param N_steer number of orientation bands
* @param N_pyr number of scales
* @return data_out ouput 2D real array
*/
static void synthesisHb(float *data_out,
                        float **list_in,
                        float **list_filters,
                        float **list_upsample_filters,
                        size_t *size_list[2],
                        size_t nx,
                        size_t ny ,
                        int N_steer,
                        int N_pyr)
{
    /* allocate memory */
    size_t sx,sy; /*number of row and columns*/
    fftwf_complex *fft_out = fftwf_malloc( nx*ny * sizeof(fftwf_complex));
    fftwf_complex *fft_list_in = fftwf_malloc(nx*ny*sizeof(fftwf_complex));
    fftwf_complex *fft_temp   = fftwf_malloc(nx*ny*sizeof(fftwf_complex));


    /* Start threaded fftw if FFTW_NTHREADS is defined */
#ifdef FFTW_NTHREADS
    if( 0 == fftwf_init_threads() )
        printf("Problem with the multi-threads initialization of fftw");
    fftwf_plan_with_nthreads(FFTW_NTHREADS);
#endif

    /* Compute the DFT of the low frequency residual */
    sx=size_list[1+N_pyr*N_steer][0];
    sy=size_list[1+N_pyr*N_steer][1];

    do_fft(fft_out, list_in[1+N_pyr*N_steer], sx, sy);

    for(int i=N_pyr-1; i>=0; i--){
        /* Upsampling */
        upsample_fft(fft_temp, fft_out, sx, sy);
        /* update current image size */
        sx *= 2;
        sy *= 2;
        /* low pass filter */
        pointwise_complexfloat_multiplication(fft_out,
                                              fft_temp,
                                              list_upsample_filters[1+i],
                                              sx*sy);
        /* Add each filtered oriented subband */
        for(int j=0; j<N_steer; j++){
            /* DFT of oriented subband */
            do_fft(fft_list_in, list_in[1+i*N_steer+j], sx, sy);
            pointwise_complexfloat_multiplication(fft_temp,
                                                  fft_list_in,
                                                  list_filters[1+i*N_steer+j],
                                                  sx*sy);
            
            /* add obtained DFT to fft_out */
            for(unsigned int i=0; i<sx*sy; i++){
                    fft_out[i][0]+=fft_temp[i][0];
                    fft_out[i][1]+=fft_temp[i][1];
            }
        }
    }
    /* last step : apply initial low frequency filter and add high frequency residual */

    /* Apply initial low frequency filter */
    pointwise_complexfloat_multiplication(fft_out,
                                          fft_out,
                                          list_upsample_filters[0],
                                          nx*ny);
    /* high pass */
    do_fft(fft_list_in, list_in[0], nx, ny);
    pointwise_complexfloat_multiplication(fft_temp,
                                          fft_list_in,
                                          list_filters[0],
                                          nx*ny);
    for(unsigned int k=0; k<nx*ny; k++){
        fft_out[k][0] += fft_temp[k][0];
        fft_out[k][1] += fft_temp[k][1];
    }

    /* final iDFT */
    do_ifft(fft_out, data_out, nx, ny);

    /* free memory */
    fftwf_free(fft_out);
    fftwf_free(fft_list_in);
    fftwf_free(fft_temp);
}

/* Texture synthesis section */

/**
* @brief Compute a texture of gray level according to the algorithm of Heeger
                & Bergen (but using the periodic component).
*
* @param data_in input array (sample texture)
* @param nxout,nyout output size
* @param nxin,nyin input size
* @param N_steer number of orientation bands
* @param N_pyr number of scales
* @param N_iteration number of iterations
* @param noise initial image
* @param list_filters_texture, list_downsample_filters_texture_filters filters
* used during the synthesis and the upsampling (texture)
* @param list_filters_sample, list_downsample_filters_sample filters used
* during the synthesis and the upsampling (sample)
* @param size_list_texture,size_list_downsample_texture list of sizes
* @param size_list_sample, size_list_downsample_sample list of sizes
* @param edge_handling option for edge handling
* @return data_out output array (synthesized texture)
*/
static void createTexture(float *data_out, float *data_in,  size_t nxout,
                          size_t nyout, size_t nxin, size_t nyin,
                          int N_steer,int N_pyr, int N_iteration, float *noise,
                          float *list_filters_sample[2+N_pyr*N_steer],
                          float *list_downsample_filters_sample[1+N_pyr],
                          size_t *size_list_sample[2+N_pyr*N_steer],
                          float *list_filters_texture[2+N_pyr*N_steer],
                          float *list_downsample_filters_texture[1+N_pyr],
                          size_t *size_list_texture[2+N_pyr*N_steer],
                          int edge_handling)
{
    /* memory allocation */

    /* for the analysis : set of images */
    float *list_sample[2+N_pyr*N_steer];
    float *list_texture[2+N_pyr*N_steer];
    float *list_sample_mirror[2+N_pyr*N_steer];

    /* for the matching */
    float *sort_values_in = malloc(2*nxin*nyin*sizeof(float));
    float *list_sort_values[2+N_pyr*N_steer];

    /* sizes */
    size_t *size_list_final_vector = malloc(
                                         2*(2+N_pyr*N_steer)*sizeof(size_t));
    size_t *size_list_final[2+N_pyr*N_steer];
    for(int k=0; k<2+N_pyr*N_steer; k++)
        size_list_final[k]=size_list_final_vector+2*k;

    /* periodic_component case */
    if(edge_handling==0) {
        for(int k=0; k<2+N_pyr*N_steer; k++) {
            size_list_final[k][0]=size_list_sample[k][0];
            size_list_final[k][1]=size_list_sample[k][1];
        }
    }
    /* mirror symmetrization case */
    else {
        for(int k=0; k<2+N_pyr*N_steer; k++) {
            size_list_final[k][0]=size_list_sample[k][0]/2;
            size_list_final[k][1]=size_list_sample[k][1]/2;
        }
    }

    /* allocate memory according to the size of the images*/
    if(edge_handling==0) {
        for(int i=0; i<2+N_pyr*N_steer; i++) {
            list_sample[i]=malloc(size_list_sample[i][0]*
                                  size_list_sample[i][1]*sizeof(float));
            list_sort_values[i]=malloc(2*size_list_sample[i][0]*
                                       size_list_sample[i][1]*sizeof(float));
        }
    } else if(edge_handling==1) {
        for(int i=0; i<2+N_pyr*N_steer; i++) {
            list_sample_mirror[i]=
                malloc(size_list_sample[i][0]*
                       size_list_sample[i][1]*sizeof(float));

            for(unsigned int m=0; m<size_list_sample[i][0]*
                    size_list_sample[i][1]; m++)
                list_sample_mirror[i][m]=0;

            list_sample[i]=malloc(size_list_final[i][0]*
                                  size_list_final[i][1]*sizeof(float));
            list_sort_values[i]=malloc(2*size_list_final[i][0]*
                                       size_list_final[i][1]*sizeof(float));
        }
    }

    /* for the synthesized texture */
    /* allocate memory */
    for(int i=0; i<2+N_pyr*N_steer; i++) {
        list_texture[i]=malloc(size_list_texture[i][0]*
                               size_list_texture[i][1]*sizeof(float));
    }


    /* initialization*/
    for(unsigned int i=0; i<nxout*nyout; i++) data_out[i]=noise[i];

    /* First histogram matching
     *(before the loop and so the analysis of the noise) */
    sort_values(sort_values_in,data_in,nxin*nyin);
    matchHist(data_out , nxout*nyout , sort_values_in , nxin*nyin);

    /* Analysis of the sample */
    if(edge_handling ==0) {
        analysisHb(list_sample, data_in, list_filters_sample,
                   list_downsample_filters_sample, size_list_sample,
                   nxin, nyin, N_steer, N_pyr);
    } else if(edge_handling==1) {
        /* compute the mirror symmetrization */
        float *data_mirror=malloc(4*nxin*nyin*sizeof(float));
        for(unsigned int i=0; i<nxin; i++) {
            for(unsigned int j=0; j<nyin; j++) {
                data_mirror[i+2*j*nxin]=data_in[i+j*nxin];
                data_mirror[i+(2*j+1)*nxin]=data_in[nxin-1-i+j*nxin];
                data_mirror[2*nyin*nxin+i+2*j*nxin]=data_in[i+(nyin-j-1)*nxin];
                data_mirror[2*nyin*nxin+i+(2*j+1)*nxin]=data_in[nxin-i-1+
                                                        (nyin-j-1)*nxin];
            }
        }

        /* analysis */
        analysisHb(list_sample_mirror, data_mirror, list_filters_sample,
                   list_downsample_filters_sample, size_list_sample,
                   2*nxin, 2*nyin, N_steer, N_pyr);

        /* keep the left-hand top corner  */
        for(int d=0; d<2+N_pyr*N_steer; d++) {
            for(unsigned int i=0; i<size_list_final[d][0]; i++) {
                for(unsigned int j=0; j<size_list_final[d][1]; j++)
                    list_sample[d][i+j*size_list_final[d][0]] =
                        list_sample_mirror[d][i+j*size_list_sample[d][0]];
            }
        }

        /* free */
        free(data_mirror);
        for(int i=0; i<2+N_pyr*N_steer; i++) free(list_sample_mirror[i]);
    }

    /* Create the list of sorted values */
    for(int s=0; s<2+N_pyr*N_steer ; s++)
        sort_values(list_sort_values[s], list_sample[s],size_list_final[s][0]*
                    size_list_final[s][1]);

    /* Loop in order to create the texture */
    for(int k=0; k<N_iteration; k++) {
        /* Create the oriented bandpass images for the texture */
        analysisHb(list_texture, data_out, list_filters_texture,
                   list_downsample_filters_texture, size_list_texture,
                   nxout, nyout, N_steer, N_pyr);

        /* Histogram matching on the list of oriented bandpass images */
        matchHistList(list_texture, size_list_texture, list_sort_values,
                      size_list_final, N_steer*N_pyr+2);

        /* Compute the texture : synthesis */
        synthesisHb(data_out,list_texture, list_filters_texture,
                    list_downsample_filters_texture, size_list_texture,
                    nxout, nyout, N_steer, N_pyr);

        /* Histogram matching (to have the same level of gray) */
        matchHist(data_out, nxout*nyout , sort_values_in, nxin*nyin);

    }

    /* free memory */
    for(int i=0; i<2+N_pyr*N_steer; i++) {
        free(list_sample[i]);
        free(list_sort_values[i]);
        free(list_texture[i]);
    }

    free(sort_values_in);
    free(size_list_final_vector);
}

/**
* @brief Compute the change of basis according to the principal
* component analysis method.
*
* @param data_in_rgb input color array
* @param nx,ny array size
* @return M_eigen_vectors the 3x3 change of basis matrix,
* data_in_rgb_mean the mean of each color component
*/
static void pca( float M_eigen_vectors[3][3], float data_in_rgb_mean[3],
                 float **data_in_rgb, size_t nx, size_t ny)
{
    float C[3][3];
    float d[3];

    /* Compute the mean of the input */
    for(int i=0; i<3; i++)    data_in_rgb_mean[i] = mean(
                    data_in_rgb[i] , nx*ny );

    /* Compute the covariance matrix C*/
    for(int i=0; i<3; i++) {
        for(int j=0; j<i+1; j++) {
            C[i][j]=0;
            for(unsigned int k=0; k<(int) nx*ny; k++) C[i][j] +=
                    (data_in_rgb[i][k] - data_in_rgb_mean[i])*
                    (data_in_rgb[j][k] - data_in_rgb_mean[j]);
            C[j][i] = C[i][j]; /* the covariance matrix is symmetric */
        }
    }

    /* Compute the change of basis matrix */
    eigen_decomposition( C, M_eigen_vectors, d);
}

/**
* @brief Compute the synthesized texture according to the algorithm of Heeger
                & Bergen (but using the periodic component)
*
* @param[in] data_in_rgb input color array (sample texture)
* @param nxout,nyout output size
* @param nxin,nyin input size
* @param N_steer number of orientation bands
* @param N_pyr number of scales
* @param N_iteration number of iterations
* @param noise initial color image
* @param edge_handling option for edge handling
* @param smooth option for the add of the smooth component
* @return data_out_rgb output color array (synthesized texture)
*/
void hb(float **data_out_rgb, float **data_in_rgb, size_t nxout,
        size_t nyout, size_t nxin, size_t nyin, int N_steer, int N_pyr,
        int N_iteration, float **noise, int edge_handling, int smooth)
{
    /* memory allocation */
    /* 3D images */

    float *data_in_pca=malloc(3 * nxin * nyin * sizeof(float));
    float *data_in_rgb_pca[3];
    data_in_rgb_pca[0] = data_in_pca;
    data_in_rgb_pca[1] = data_in_pca + nxin * nyin;
    data_in_rgb_pca[2] = data_in_pca + 2 * nxin * nyin;

    float *data_out_pca=malloc(3 * nxout * nyout * sizeof(float));
    float *data_out_rgb_pca[3];
    data_out_rgb_pca[0] = data_out_pca;
    data_out_rgb_pca[1] = data_out_pca + nxout * nyout;
    data_out_rgb_pca[2] = data_out_pca + 2 * nxout * nyout;

    float *data_in_per=malloc(3 * nxin * nyin * sizeof(float));
    float *data_in_rgb_per[3];
    data_in_rgb_per[0] = data_in_per;
    data_in_rgb_per[1] = data_in_per + nxin * nyin;
    data_in_rgb_per[2] = data_in_per + 2 * nxin * nyin;

    /* PCA */
    float data_in_rgb_mean[3];
    float M_eigen_vectors[3][3];

    /* Compute the periodic component of the input */
    if(edge_handling == 0) {
        for(int i=0; i<3 ; i++ )
            periodic_component(data_in_rgb_per[i],data_in_rgb[i],nxin,nyin);
    } else {
        for(int d=0; d<3; d++) {
            for(unsigned int i=0; i<nxin*nyin; i++)
                data_in_rgb_per[d][i] = data_in_rgb[d][i];
        }
    }

    /* Make the change of basis RGB--> PCA basis */
    pca( M_eigen_vectors, data_in_rgb_mean , data_in_rgb_per, nxin , nyin );

    for(int i=0; i<3; i++) {
        for(unsigned int j=0; j< nxin*nyin; j++) {
            data_in_rgb_pca[i][j]=0;
            for(int k=0; k<3; k++) data_in_rgb_pca[i][j] +=
                    M_eigen_vectors[k][i]*
                    (data_in_rgb_per[k][j]-data_in_rgb_mean[k]);
        }
    }

    /* Compute the set of filters */
    /* define the factor of reversibility alpha */
    float alpha;
    alpha = compute_alpha(N_steer);

    /* memory allocation for the filters and their sizes */
    float *list_filters_sample[2+N_pyr*N_steer];
    float *list_downsample_filters_sample[1+N_pyr];

    size_t *size_list_sample_vector = malloc(
                                          2*(2+N_pyr*N_steer)*sizeof(size_t));
    size_t *size_list_downsample_sample_vector =
        malloc(2*(1+N_pyr)*sizeof(size_t));

    size_t *size_list_sample[2+N_pyr*N_steer];
    for(int k=0; k<2+N_pyr*N_steer; k++) size_list_sample[k]=
            size_list_sample_vector+2*k;
    size_t *size_list_downsample_sample[1+N_pyr];
    for(int k=0; k<1+N_pyr; k++) size_list_downsample_sample[k]=
            size_list_downsample_sample_vector+2*k;

    float *list_filters_texture[2+N_pyr*N_steer];
    float *list_downsample_filters_texture[1+N_pyr];

    size_t *size_list_texture_vector = malloc(
                                           2*(2+N_pyr*N_steer)*sizeof(size_t));
    size_t *size_list_downsample_texture_vector =
        malloc(2*(1+N_pyr)*sizeof(size_t));

    size_t *size_list_texture[2+N_pyr*N_steer];
    for(int k=0; k<2+N_pyr*N_steer; k++)
        size_list_texture[k]=size_list_texture_vector+2*k;
    size_t *size_list_downsample_texture[1+N_pyr];
    for(int k=0; k<1+N_pyr; k++) size_list_downsample_texture[k]=
            size_list_downsample_texture_vector+2*k;

    /* Create the filters and the lists of sizes */

    /* for the sample */
    if(edge_handling == 0) {
        size_filters(size_list_sample , size_list_downsample_sample ,
                     nxin , nyin , N_steer, N_pyr);
    } else {
        size_filters(size_list_sample , size_list_downsample_sample ,
                     2*nxin , 2*nyin , N_steer, N_pyr);
    }

    /* allocate memory */
    for(int i=0; i<2+N_pyr*N_steer; i++) list_filters_sample[i]=
            malloc(size_list_sample[i][0]*
                   size_list_sample[i][1]*sizeof(float));
    for(int i=0; i<1+N_pyr; i++) {
        list_downsample_filters_sample[i]=
            malloc(
                size_list_downsample_sample[i][0]*
                size_list_downsample_sample[i][1]*
                sizeof(float));
    }

    if(edge_handling == 0) {
        filters(list_filters_sample, list_downsample_filters_sample ,
                size_list_sample ,
                size_list_downsample_sample , nxin , nyin ,
                N_steer, N_pyr, alpha);
    } else {
        filters(list_filters_sample, list_downsample_filters_sample ,
                size_list_sample ,
                size_list_downsample_sample , 2*nxin , 2*nyin ,
                N_steer, N_pyr, alpha);
    }

    size_filters(size_list_texture , size_list_downsample_texture ,
                 nxout , nyout , N_steer, N_pyr);

    /* allocate memory */
    for(int i=0; i<2+N_pyr*N_steer; i++) {
        list_filters_texture[i]=
            malloc(size_list_texture[i][0]*
                   size_list_texture[i][1]*
                   sizeof(float));
    }
    for(int i=0; i<1+N_pyr; i++) {
        list_downsample_filters_texture[i]=
            malloc(size_list_downsample_texture[i][0]*
                   size_list_downsample_texture[i][1]*sizeof(float));
    }

    filters(list_filters_texture, list_downsample_filters_texture ,
            size_list_texture , size_list_downsample_texture ,
            nxout , nyout , N_steer, N_pyr, alpha);

    /* Compute the texture for every color component */
    for(int i=0; i<3; i++) createTexture( data_out_rgb_pca[i],
                                              data_in_rgb_pca[i], nxout, nyout,
                                              nxin, nyin, N_steer, N_pyr,
                                              N_iteration , noise[i],
                                              list_filters_sample,
                                              list_downsample_filters_sample,
                                              size_list_sample,
                                              list_filters_texture,
                                              list_downsample_filters_texture,
                                              size_list_texture,
                                              edge_handling);

    /* Make the invert change of basis PCA --> RGB */
    for(int i=0; i<3; i++) {
        for(unsigned int j=0; j<nxout*nyout; j++) {
            data_out_rgb[i][j]  =  data_in_rgb_mean[i];
            for(int k=0; k<3; k++) data_out_rgb[i][j] += M_eigen_vectors[i][k]*
                        data_out_rgb_pca[k][j];
        }
    }

    /* Smooth component part */
    if(smooth == 1) {
        /* Make a zoom if there is an extension */
        if (nxin == nxout && nyin == nyout) {
            for(int i=0; i<3; i++) {
                for(unsigned int j=0; j<nxout*nyout; j++) data_out_rgb[i][j]+=
                        data_in_rgb[i][j]-data_in_rgb_per[i][j];
            }
        } else {
            float *data_in_smooth=malloc(3 * nxin * nyin * sizeof(float));
            float *data_in_smooth_rgb[3];
            data_in_smooth_rgb[0] = data_in_smooth;
            data_in_smooth_rgb[1] = data_in_smooth + nxin * nyin;
            data_in_smooth_rgb[2] = data_in_smooth + 2 * nxin * nyin;

            float *data_zoom_smooth=malloc(3 * nxout * nyout * sizeof(float));
            float *data_zoom_smooth_rgb[3];
            data_zoom_smooth_rgb[0] = data_zoom_smooth;
            data_zoom_smooth_rgb[1] = data_zoom_smooth + nxout * nyout;
            data_zoom_smooth_rgb[2] = data_zoom_smooth + 2 * nxout * nyout;

            for(int i=0; i<3; i++) {
                for(unsigned int j=0; j<nxout*nyout; j++)
                    data_in_smooth_rgb[i][j] =
                        data_in_rgb[i][j]-data_in_rgb_per[i][j];
            }

            for(int i=0; i<3; i++) {
                zoom_bilin(data_zoom_smooth_rgb[i], nxout,
                           nyout,data_in_smooth_rgb[i],  nxin, nyin);
                for(unsigned int j=0; j<nxout*nyout; j++)
                    data_out_rgb[i][j] += data_zoom_smooth_rgb[i][j];
            }

            free(data_in_smooth);
            free(data_zoom_smooth);
        }

        /* Final histogram matching in the RGB base */
        float *sort_values_final = malloc(2*nxin*nyin*sizeof(float));
        for(int i=0; i<3; i++) {
            sort_values(sort_values_final, data_in_rgb[i],nxin*nyin);
            matchHist(data_out_rgb[i], nxout*nyout,
                      sort_values_final, nxin*nyin);
        }
        free(sort_values_final);
    } else {
        /* Final histogram matching in the RGB base */
        float *sort_values_final = malloc(2*nxin*nyin*sizeof(float));
        for(int i=0; i<3; i++) {
            sort_values(sort_values_final,data_in_rgb_per[i],nxin*nyin);
            matchHist(data_out_rgb[i], nxout*nyout,
                      sort_values_final, nxin*nyin);
        }
        free(sort_values_final);
    }

    /* free */
    free(size_list_sample_vector);
    free(size_list_downsample_sample_vector);
    free(size_list_texture_vector);
    free(size_list_downsample_texture_vector);
    free(data_in_pca);
    free(data_in_per);
    free(data_out_pca);

    for(int i=0; i<2+N_pyr*N_steer; i++) {
        free(list_filters_sample[i]);
        free(list_filters_texture[i]);
    }
    for(int i=0; i<1+N_pyr; i++) {
        free(list_downsample_filters_sample[i]);
        free(list_downsample_filters_texture[i]);
    }

}
