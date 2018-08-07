/* matching_hist.c contains the function for the histogram matching */

void matchHist(float *im_out, size_t length_out, float *sort_values_ref,
               size_t length_ref);
void matchHistList(float **list_im_out,size_t *size_out[2],
                   float **list_sort_values_in,size_t *size_in[2],
                   int length_list);
void sort_values(float *sort_values_out, float *im_in, size_t N);
