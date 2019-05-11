#include <stddef.h>
void _prod(double *omd, long *not_obs, size_t not_obs_size, long *V, double *num);

void _prod(double *omd, long *not_obs, size_t not_obs_size, long *V, double *num) 
{
    size_t i, j, count = 0, omd_indx, v_size, v_indx;
    
    for (i = 0; i < not_obs_size; ++i){
        v_indx = not_obs[i];
        v_size = V[v_indx];
        for (j = 0; j < v_size; ++j){
            omd_indx = count + j;
            num[i] *= omd[omd_indx];
        };
        count += v_size;
    };
}
