/* hb_lib.c contains the main function for the texture synthesis */

void hb(float **data_out_rgb, float **data_in_rgb,
        size_t nxout, size_t nyout, size_t nxin, size_t nyin, int N_steer,
        int N_pyr, int N_iteration, float **noise,
        int edge_handling, int smooth);
