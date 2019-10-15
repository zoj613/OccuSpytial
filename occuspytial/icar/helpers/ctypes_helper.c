#include <stddef.h>
void _proba(double* omd, size_t* not_obs, size_t not_obs_size, size_t* V, double* occ_proba);

void _proba(double* omd, size_t* not_obs, size_t not_obs_size, size_t* V, double* occ_proba) 
{
    size_t count = 0;
    double old_occ_proba;

    for (size_t i = 0; i < not_obs_size; i++){
        old_occ_proba = occ_proba[i];
        for (size_t j = 0; j < V[not_obs[i]]; j++){
            occ_proba[i] *= omd[count + j];
        };
        occ_proba[i] /= (1. - old_occ_proba + occ_proba[i]);
        count += V[not_obs[i]];
    };
}
