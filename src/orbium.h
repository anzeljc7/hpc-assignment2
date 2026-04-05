#ifndef ORBIUM_H
#define ORBIUM_H
#include <stdint.h>
#define ORBIUM_SIZE 20

#ifdef __cplusplus
extern "C"
{
#endif

    extern uint8_t inferno_pallete[];
    double *place_orbium(double *world, unsigned int rows, unsigned int cols, unsigned int x, unsigned int y, unsigned int angle);

#ifdef __cplusplus
}
#endif

#endif
