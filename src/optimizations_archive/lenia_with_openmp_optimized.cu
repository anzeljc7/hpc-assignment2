#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

#define GENERATE_GIF
#define w(r, c) (w[(r) * kernel_size + (c)])

// OPTIMIZACIJA: Uporaba float in expf za hitrejše računanje na CPE
inline float gauss(float x, float mu, float sigma)
{
    float diff = (x - mu) / sigma;
    return expf(-0.5f * (diff * diff));
}

float growth_lenia(float u)
{
    float mu = 0.15f;
    float sigma = 0.015f;
    return -1.0f + 2.0f * gauss(u, mu, sigma);
}

float *generate_kernel(float *K, const unsigned int size)
{
    float mu = 0.5f;
    float sigma = 0.15f;
    int r = size / 2;
    float sum = 0;
    if (K != NULL)
    {
        for (int y = -r; y < r; y++)
        {
            for (int x = -r; x < r; x++)
            {
                // Uporaba sqrtf za float natančnost
                float distance = sqrtf((float)((1 + x) * (1 + x) + (1 + y) * (1 + y))) / r;
                if (distance > 1.0f)
                    K[(y + r) * size + x + r] = 0.0f;
                else
                    K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                sum += K[(y + r) * size + x + r];
            }
        }
        for (unsigned int i = 0; i < size * size; i++)
            K[i] /= sum;
    }
    return K;
}

float *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const float dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));
    float *next_world = (float *)calloc(rows * cols, sizeof(float));

    w = generate_kernel(w, kernel_size);

    // POMEMBNO: Če place_orbium v orbium.c uporablja double, bo tole še vedno problem.
    // Za testiranje CPE baseline-a uporabi double, če place_orbium ne moreš spremeniti.
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        world = (float *)place_orbium((double *)world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    int r_k = kernel_size / 2;

    for (unsigned int step = 0; step < steps; step++)
    {
// OpenMP paralelizacija po vrsticah [cite: 3, 39]
#pragma omp parallel for schedule(static)
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                float sum = 0;
                // Konvolucija [cite: 21]
                for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    int ii = (i - r_k + kri + rows) % rows;
                    for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        int jj = (j - r_k + kcj + cols) % cols;
                        // POPRAVEK: i * cols + j namesto i * rows + j
                        sum += w[ki * kernel_size + kj] * world[ii * cols + jj];
                    }
                }

                // Loop Fusion: Konvolucija in evolucija sta združeni
                float res = world[i * cols + j] + dt * growth_lenia(sum);

                if (res > 1.0f)
                    res = 1.0f;
                else if (res < 0.0f)
                    res = 0.0f;

                next_world[i * cols + j] = res;

#ifdef GENERATE_GIF
                gif->frame[i * cols + j] = (uint8_t)(res * 255);
#endif
            }
        }

        // Pointer Swap: Hitra zamenjava matrik
        float *temp = world;
        world = next_world;
        next_world = temp;

#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    free(w);
    free(next_world);
    return world;
}