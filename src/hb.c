/*
 * Copyright(C) 2013, Thibaud Briand, ENS Cachan <thibaud.briand@ens-cachan.fr>
 * Copyright(C) 2013, Jonathan Vacher, ENS Cachan <jvacher@ens-cachan.fr>
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
* @file hb.c
* @brief Main function for Heeger & Bergen texture synthesis.
* Important modules are in the file @ref hb_lib.c
*
* @version 1.1
* @author Thibaud BRIAND & Jonathan VACHER ;
* <thibaud.briand@ens-cachan.fr> ; <jvacher@ens-cachan.fr>
*/

/**
* @mainpage Source code for Heeger & Bergen texture synthesis
*
* @image html hb_description.png "Random phase texture synthesis using ./hb"
*
* This source code is an ANSI C implementation of the Heeger & Bergen
* texture synthesis algorithm described in the IPOL webpage
* <a href="http://adresse_pipo.com"
* title="Go to the IPOL webpage Heeger & Bergen Synthesis">
* Heeger & Bergen Synthesis</a>.
*
* The last release should be available at <http://adresse_pipo2.com/>.
*
* This code is provided with an HTML documentation produced
* by doxygen <http://adresse_pipo3.com/>.
* 
* The @ref main function only deals with input/output. The function of main interest is @ref hb.
*
* @author Thibaud BRIAND & Jonathan VACHER ;
* <thibaud.briand@ens-cachan.fr> ; <jvacher@ens-cachan.fr>
*
* @b Requirements: @n
* - ANSI C compiler
* - getopt
* - libpng
* - libfftw3
* - Gomp (optional)
*
* @b Compilation: @n
*
* Execute the provided Makefile (function make).
*
* @b Usage: (displayed in executing ./hb -h or see @ref hb.c)@n
*
*  hb -s scales -k orientations -i iterations -n noise.png
* -g seed -x row_ratio -y colum_ratio -e edge_handling -r smooth
* -p crop.png input.png output.png @n
*
*       <em> Required parameters:</em> @n
*
* @li input.png   :   name of the input PNG image @n
* @li output.png  :   name of the output PNG image @n
*
*       <em> Optionnal parameters:</em> @n
*
* @li scales         :   int to specify the number of pyramid scales
*                        (by default 4) @n
* @li orientations   :   int>=2 to specify the number of orientations
*                        (by default 4) @n
* @li iterations     :   int>=1 to specify the number of iterations
*                        (by default 5) @n
* @li noise          :   name of the noise PNG image @n
* @li seed           :   unsigned int to specify the seed for
*                        the random number generator (seed = time(NULL)
*                        by default) @n
* @li row_ratio      :   int>=1 to specify the row ratio of extension
*                        (by default 1) @n
* @li column_ratio   :   int>=1 to specify the column ratio of extension
*                        (by default 1) @n
* @li edge_handling  :   int to specify the type of edge handling :
*                        periodic component (0) and mirror symmetrization (1)@n
* @li smooth         :   int=1 to add the smooth component (by default 0) @n
* @li crop           :   int=1 to write the cropped input
*                        (PNG image called input_cropped.png) @n
*
* @b Test:
*
* To test the module run: @n
*    ./hb -g 0 data/sample.png output.png @n
* output.png should be the same as the imacge sample_hb.png provided
* with the source code.
*
* @b Releases :
*
* @li Version 0.98 : Beta version.
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <fftw3.h>
#include "io_png.h"
#include "hb_lib.h"
#include "mt19937ar.h"

/** print a message and abort the execution */
#define FATAL(MSG)			\
do {				\
	fprintf(stderr, MSG "\n");	\
	abort();			\
	} while (0);

/*
* Main function section
*/

/**
* @brief Display the usage of the main module on stdout.
*/

static int display_usage()
{
    printf("\nhb -s scales -k orientations -i iterations -n noise.png\n");
    printf("-g seed -x row_ratio -y colum_ratio -e edge_handling -r smooth\n");
    printf("-p crop input.png output.png \n\n");
    printf("Required parameters:\n");
    printf("   input.png   :   name of the input PNG image\n");
    printf("   output.png  :   name of the output PNG image\n\n");
    printf("Optionnal parameters:\n");
    printf("   scales          :   int to specify the number of pyramid\n");
    printf("                       scales (by default 4)\n");
    printf("   orientations    :   int>=2 to specify the number of\n");
    printf("                       orientations (by default 4)\n");
    printf("   iterations      :   int>=1 to specify the number of\n");
    printf("                       iterations (by default 5)\n");
    printf("   noise           :   name of the noise PNG image\n");
    printf("   seed            :   unsigned int to specify the seed for\n");
    printf("                       the random number generator\n");
    printf("                       (seed = time(NULL) by default)\n");
    printf("   row_ratio       :   int>=1 to specify the row ratio of\n");
    printf("                       extension (by default 1)\n");
    printf("   column_ratio    :   int>=1 to specify the column ratio of\n");
    printf("                       extension (by default 1)\n");
    printf("   edge_handling   :   int to specify type of edge handling\n");
    printf("                       periodic component (0) \n");
    printf("                       and mirror symmetrization (1)\n");
    printf("   smooth          :   int=1 to add the smooth component\n");
    printf("                       (by default 0)\n");
    printf("   crop            :   int=1 to write the cropped input\n");
    printf("                       (PNG image called input_cropped.png)\n");
    return(1);
}


/**
* @brief Main function call (input-output and options handling)
*/
int main(int argc, char **argv)
{
    char *fname_in;                 /* input/output/crop file names */
    char *fname_out;

    float *data_in;                 /*input data */
    float *data_in_rgb[3];
    float *data_in_crop;		/* to crop the input */
    float *data_in_crop_rgb[3];
    float *data_out;                /* output data */
    float *data_out_rgb[3];
    float *noise_in= {0};               /* noise data */
    float *noise_in_rgb[3];
    float *noise_in_uncrop= {0};      /* noise data */
    float *noise_in_uncrop_rgb[3]= {0};

    /* input sizes */
    size_t nx_in=0;
    size_t ny_in=0;
    size_t nxin=0;
    size_t nyin=0;
    /* noise and output sizes */
    size_t nx_out=0;
    size_t ny_out=0;
    size_t nxout=0;
    size_t nyout=0;

    /* "Default" value initialization */
    int N_steer=4;		/* Number of orientation bands */
    int N_pyr=4;		/* Number of pyramid levels (scales) */
    int N_iteration=5;  	/* Number of iterations */
    int factor_noise=0;     /* optional noise */
    int c;			/* getopt */
    int rx=1;		/* row extension */
    int ry=1;		/* column extension */
    int edge_handling=0;
    int smooth=0;
    int crop=0;
    unsigned long seed = time(NULL); /* seed for the noise */

    /* process the options and parameters */

    while ((c = getopt (argc, argv, "s:k:i:n:g:x:y:r:e:p:hv")) != -1) {
        switch (c) {
        case 's':
            /* Number of scales specified */
            N_pyr = (int) atoi(optarg);
            if(N_pyr <0)
                FATAL("Number of scales must be greater than 0.")
                break;
        case 'k':
            /* Number of orientations specified */
            N_steer = (int) atoi(optarg);
            if(N_steer <2)
                FATAL("Number of orientations must be greater than 2.")
                break;
        case 'i':
            /* Number of iterations */
            N_iteration = (int) atoi(optarg);
            if(N_iteration <1)
                FATAL("Number of iteration must be greater than 1.")
                break;
        case 'n':
            /* noise specified */
            factor_noise = 1; /* optional noise is off */
            char *fname_noise = argv[optind-1];
            /* read the input noise */
            if(NULL == (noise_in_uncrop = io_png_read_f32_rgb(
                                              fname_noise, &nx_out, &ny_out)))
                FATAL("error while reading the PNG noise");
            noise_in_uncrop_rgb[0] = noise_in_uncrop;
            noise_in_uncrop_rgb[1] = noise_in_uncrop + nx_out * ny_out;
            noise_in_uncrop_rgb[2] = noise_in_uncrop + 2 * nx_out * ny_out;
            break;
        case 'g':
            /* seed specified */
            seed = (unsigned long) atoi(optarg);
            break;
        case 'x':
            /*row ratio specified */
            rx = (int) atoi(optarg);
            if(rx<1)
                FATAL("The row ratio must be an integer greater than 1.")
                break;
        case 'y':
            /*column ratio specified */
            ry = (int) atoi(optarg);
            if(ry<1)
                FATAL("The column ratio must be an integer greater than 1.")
                break;
        case 'e':
            /* edge_handling specified */
            edge_handling = (int) atoi(optarg);
            if(edge_handling > 1)
                FATAL("Unknown edge handling option.")
                break;
        case 'r':
            /* smooth component specified */
            smooth = (int) atoi(optarg);
            if(smooth > 1)
                FATAL("Set smooth = 0 or 1 to use the smooth option.")
                break;
        case 'p':
            /* crop specified */
            crop = 1;
            break;
        case 'h':
            /* display usage */
            display_usage();
            return(-1);
        case 'v':
            /* display version */
            fprintf(stdout, "%s version " __DATE__ "\n", argv[0]);
            return(-1);

        default:
            abort();
        }
    }

    /* odd case */
    if(smooth*edge_handling>0)
        FATAL("Smooth component and mirror symmetrization incompatible");

    /* process the non-option parameters */
    if (2 > (argc - optind)) {
        printf("The image file names are missing\n\n");
        display_usage();
        return(-1);
    }
    fname_in = argv[optind++];
    fname_out = argv[optind++];

    /* for generating noise */
    mt_init_genrand(seed); /* seed */

    /* read the PNG image and set pointers for data_in */
    if (NULL == (data_in = io_png_read_f32_rgb(fname_in, &nx_in, &ny_in)))
        FATAL("error while reading the PNG image");
    data_in_rgb[0] = data_in;
    data_in_rgb[1] = data_in + nx_in * ny_in;
    data_in_rgb[2] = data_in + 2 * nx_in * ny_in;

    /* crop the input to be able to compute the N_pyr scales */
    /* row */
    int     rem_nx = (nx_in)%((1<<N_pyr));
    int     r_nx = (rem_nx/2);
    nxin = nx_in - rem_nx;
    if(nxin<=0) FATAL("Horizontal input size must be greater than 2^N_pyr")

        /* column */
        int rem_ny = (ny_in)%(1<<N_pyr);
    int    r_ny = (rem_ny/2);
    nyin = ny_in - rem_ny;
    if(nyin<=0) FATAL("Vertical input size must be greater than 2^N_pyr")

        /* set pointers for data_in_crop */

        if( NULL == (data_in_crop = (float *)
                                    malloc(3*nxin*nyin*sizeof(float))))
            FATAL("Allocation error");
    data_in_crop_rgb[0] = data_in_crop;
    data_in_crop_rgb[1] = data_in_crop + nxin * nyin;
    data_in_crop_rgb[2] = data_in_crop + 2 * nxin * nyin;

    /* compute the cropped image */
    for(int k=0; k<3; k++) {
        for(unsigned int i=0; i<nxin; i++) {
            for(unsigned int j=0; j<nyin; j++) {
                data_in_crop_rgb[k][i+j*nxin]  =
                    data_in_rgb[k][i+r_nx+(j+r_ny)*nx_in];
            }
        }
    }

    /* write a PNG image from the cropped input */
    if(crop==1) {
        if (0 != io_png_write_f32("input_cropped.png",
                                  data_in_crop, nxin, nyin, 3))
            FATAL("error while writing the input cropped PNG file");
    }

    /* set the noise */
    if( factor_noise == 1) {
        /* noise size and possible crop*/
        int r_out_nx = (nx_out)%(nxin);
        nxout = nx_out - r_out_nx;
        if(nxout<=0) FATAL("Horizontal noise size smaller than the input one")
            int r_out_ny = (ny_out)%(nyin);
        nyout = ny_out - r_out_ny;
        if(nyout<=0) FATAL("Vertical noise size smaller than the input one")

//             /* set pointers for the uncropped noise */
//             if( NULL == (noise_in_uncrop = (float *)
//                                            malloc(
//                                                3*nx_out*ny_out*sizeof(float))))
//                 FATAL("Allocation error");

            /* set pointers for the cropped noise */
            if( NULL == (noise_in = (float *)
                                    malloc(3 * nxout * nyout * sizeof(float))) )
                FATAL("Allocation error");
        noise_in_rgb[0] = noise_in;
        noise_in_rgb[1] = noise_in + nxout * nyout;
        noise_in_rgb[2] = noise_in + 2 * nxout * nyout;

        /* compute the cropped noise */
        for(int k=0; k<3; k++) {
            for(unsigned int i=0; i<nxout; i++) {
                for(unsigned int j=0; j<nyout; j++) {
                    noise_in_rgb[k][i+j*nxout]=(float)
                                               noise_in_uncrop_rgb[k]
                                               [i+r_out_nx/2+
                                                (j+r_out_ny/2)*nx_out];
                }
            }
        }
        free(noise_in_uncrop);
    } else if(factor_noise ==0) {
        /* create a white noise */
        /* noise sizes */
        nxout = rx*nxin;
        nyout = ry*nyin;

        /* set pointers for the noise */
        if( NULL == (noise_in = (float *)
                                malloc(3 * nxout * nyout * sizeof(float))) )
            FATAL("Allocation error");
        noise_in_rgb[0] = noise_in;
        noise_in_rgb[1] = noise_in + nxout * nyout;
        noise_in_rgb[2] = noise_in + 2 * nxout * nyout;

        for(int k=0; k<3; k++) {
            for(unsigned int i=0; i<nxout*nyout; i++) {
                noise_in_rgb[k][i]  =  (float) 255*mt_genrand_res53();
            }
        }
    }

    /* Computation of the texture */

    /* set pointers for data_out */
    if( NULL == (data_out = (float *)
                            malloc(3 * nxout * nyout * sizeof(float))) )
        FATAL("allocation error");
    data_out_rgb[0] = data_out;
    data_out_rgb[1] = data_out +  nxout * nyout;
    data_out_rgb[2] = data_out + 2 * nxout * nyout;

    /* HB computation */
    hb(data_out_rgb, data_in_crop_rgb,
       nxout, nyout, nxin, nyin, N_steer, N_pyr,
       N_iteration, noise_in_rgb,edge_handling,smooth);

    /* Write result in PNG file */
    if (0 != io_png_write_f32(fname_out, data_out, nxout, nyout, 3))
        FATAL("error while writing output PNG file");

    /* Free memory */
    free(data_in);
    free(data_in_crop);
    free(noise_in);
    free(data_out);

    return(0);
}
