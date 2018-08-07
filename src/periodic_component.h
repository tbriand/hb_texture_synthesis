/* periodic_component.c contains the function for computing
 * the periodic component
 */

fftwf_complex *pointwise_complexfloat_multiplication(fftwf_complex *comp_out,
        fftwf_complex *comp_in,
        float *float_in, size_t N);
float *periodic_component(float *data_in_per, float *data_in, size_t nx, size_t ny);
float mean(float *data_in, size_t N);
