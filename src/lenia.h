#ifndef LENIA_H
#define LENIA_H

#ifdef __cplusplus
extern "C" {
#endif

struct orbium_coo { 
    int row;
    int col;
    int angle;
};

double *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums);

#ifdef __cplusplus
}
#endif

#endif 


