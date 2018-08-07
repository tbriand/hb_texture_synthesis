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
* @file bilinear_zoom.c
* @brief Compute the bilinear zoom of an image.
*
* @version 0.98
* @author Thibaud BRIAND & Jonathan VACHER ; <thibaud.briand@ens-cachan.fr> ;
* <jvacher@ens-cachan.fr>
*
* @note Useful if you add the smooth component after using the periodic
* component option (see @ref periodic_component.c file and @ref hb function).
*/

/* Bilinear zoom section */

/**
 * @brief Get the value of a pixel array.
 *
 * @param[in] float *x input array
 * @param int w,h input sizes
 * @param int i,j index
 * @return float x[i + j*w] pixel corresponding
 */
static float getpixel(float *x, int w, int h, int i, int j)
{
    if (i < 0 || i >= w || j < 0 || j >= h)
        return 0;

    return x[i + j*w];
}

/**
 * @brief Set the value of a pixel array.
 *
 * @param[in] float *x input array
 * @param int w,h input sizes
 * @param int i,j index
 * @param float v pixel value
 * @return set the value of x[i + j*w] to v;
 */
static void setpixel(float *x, int w, int h, int i, int j, float v)
{
    if (i < 0 || i >= w || j < 0 || j >= h)
        return;

    x[i + j*w] = v;
}


/**
 * @brief Evaluate the value of the bilinear zoom
 *
 * @param[in] float a,b,c,d
 * @param float x,y
 * @return float r the value of the bilinear zoom at the considered pixel
 */
static float evaluate_bilinear(float a, float b, float c, float d,float x,
                               float y)
{
    float r = 0;

    r += a * (1-x) * (1-y);
    r += b * ( x ) * (1-y);
    r += c * (1-x) * ( y );
    r += d * ( x ) * ( y );

    return r;
}

/**
* @brief Compute the bilinear zoom of a float array.
*
* @param[in] float *x input array
* @param int w,h input sizes
* @param int W,H output sizes
* @return float *X bilinear zoom of the array x
*/
void zoom_bilin(float *X, int W, int H, float *x, int w, int h)
{
    /* set ratio of zoom */
    float wfactor = w/(float)W;
    float hfactor = h/(float)H;

    /* loop on the whole output array */
    for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++) {
            float p = i*wfactor;
            float q = j*hfactor;

            int ip = p;
            int iq = q;

            float a = getpixel(x, w, h, ip  , iq  );
            float b = getpixel(x, w, h, ip+1, iq  );
            float c = getpixel(x, w, h, ip  , iq+1);
            float d = getpixel(x, w, h, ip+1, iq+1);
            float r = evaluate_bilinear(a, b, c, d, p-ip, q-iq);

            setpixel(X, W, H, i, j, r);
        }
}
