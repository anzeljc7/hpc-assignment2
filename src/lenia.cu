#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

#define GENERATE_GIF

// Makro za dostop do jedra
#define w(r, c) (w[(r) * kernel_size + (c)])

// OPTIMIZACIJA 1: Uporaba množenja namesto funkcije pow() in float natančnosti [cite: 35, 37]
// Funkcija pow() je počasna; množenje (diff * diff) procesor opravi v enem ciklu.
inline float gauss(float x, float mu, float sigma)
{
    float diff = (x - mu) / sigma;
    return expf(-0.5f * (diff * diff)); // expf() je hitrejša float različica
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
    int r = (int)size / 2;
    float sum = 0;
    if (K != NULL)
    {
        for (int y = -r; y < r; y++)
        {
            for (int x = -r; x < r; x++)
            {
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

    // OPTIMIZACIJA 2: Dvojno bufferiranje (Pointer Swap)
    // Namesto nenehne alokacije tmp tabele, pripravimo world in next_world vnaprej.
    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));
    float *next_world = (float *)calloc(rows * cols, sizeof(float));

    w = generate_kernel(w, kernel_size);

    // Inicializacija orbiumov (funkcija place_orbium uporablja double, zato d_world) [cite: 19]
    double *d_world = (double *)calloc(rows * cols, sizeof(double));
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        d_world = place_orbium(d_world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }
    for (unsigned int i = 0; i < rows * cols; i++)
        world[i] = (float)d_world[i];
    free(d_world);

    int r_k = (int)kernel_size / 2;

    for (unsigned int step = 0; step < steps; step++)
    {
// OPTIMIZACIJA 3: OpenMP paralelizacija (razdelitev po vrsticah) [cite: 3, 32, 39]
#pragma omp parallel for schedule(static)
        for (unsigned int i = 0; i < rows; i++)
        {
            // OPTIMIZACIJA 4: Odprava operatorja % za vrstice
            // Indekse sosednjih vrstic izračunamo le enkrat na začetku zanke 'i'.
            int row_indices[128];
            for (int ki = 0; ki < (int)kernel_size; ki++)
            {
                int ii = (int)i - r_k + ki;
                if (ii < 0)
                    ii += (int)rows;
                else if (ii >= (int)rows)
                    ii -= (int)rows;
                row_indices[ki] = ii * (int)rows; // Uporabljamo tvoj stride 'rows'
            }

            for (unsigned int j = 0; j < cols; j++)
            {
                float sum = 0;
                // KONVOLUCIJA: Zrcaljenje jedra (Flipping) [cite: 21]
                for (int ki = (int)kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    int row_offset = row_indices[kri];
                    for (int kj = (int)kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        // OPTIMIZACIJA 5: Odprava operatorja % za stolpce
                        // Uporaba preprostega 'if' stavka je na CPU bistveno hitrejša od modula.
                        int jj = (int)j - r_k + kcj;
                        if (jj < 0)
                            jj += (int)cols;
                        else if (jj >= (int)cols)
                            jj -= (int)cols;

                        sum += w[ki * kernel_size + kj] * world[row_offset + jj];
                    }
                }

                // OPTIMIZACIJA 6: Združevanje zank (Loop Fusion)
                // Izračun rasti in posodobitev izvedemo takoj, da zmanjšamo promet s pomnilnikom.

                // DODATNA OPTIMIZACIJA: Preskok računanja Gause pri u=0
                float res;
                if (sum < 1e-6f)
                {
                    res = world[i * (int)rows + j] - dt;
                }
                else
                {
                    res = world[i * (int)rows + j] + dt * growth_lenia(sum);
                }

                if (res > 1.0f)
                    res = 1.0f;
                else if (res < 0.0f)
                    res = 0.0f;

                next_world[i * (int)rows + j] = res;

#ifdef GENERATE_GIF
                gif->frame[i * (int)rows + j] = (uint8_t)(res * 255.0f);
#endif
            }
        }

        // OPTIMIZACIJA 7: Zamenjava kazalcev (Pointer Swap)
        // Namesto kopiranja matrike v vsakem koraku le zamenjamo naslova v pomnilniku.
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