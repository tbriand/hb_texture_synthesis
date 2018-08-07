/* filters.c contains the functions used for computing the filters */

void size_filters(size_t **size_list, size_t **size_list_downsample,
                  size_t nx,size_t ny ,int N_steer, int N_pyr);
void filters(float **list_filters, float **list_downsample_filters,
             size_t *size_list[2], size_t *size_list_downsample[2],
             size_t nx,size_t ny ,int N_steer, int N_pyr, float alpha);
float compute_alpha(int N_steer);
