#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

#define GENERATE_GIF

// Makro za dostop do elementa jedra po vrstici in stolpcu
#define w(r, c) (w[(r) * kernel_size + (c)])

/*
 * OPTIMIZACIJA 1: Gaussova funkcija z množenjem namesto pow()
 *
 * Klic pow(x, 2.0) je procesorsko zahteven, ker uporablja splošno
 * eksponentno rutino. Neposredno množenje (diff * diff) procesor
 * opravi v enem ciklu. Uporabimo tudi expf() – hitrejšo, za float
 * optimizirano različico funkcije exp().
 */
inline float gauss(float x, float mu, float sigma)
{
    float diff = (x - mu) / sigma;
    return expf(-0.5f * (diff * diff));
}

/*
 * Funkcija rasti (growth function) za Lenio.
 */
float growth_lenia(float u)
{
    float mu = 0.15f;
    float sigma = 0.015f;
    return -1.0f + 2.0f * gauss(u, mu, sigma);
}

/*
 * Generiranje konvolucijskega jedra (kernel).
 *
 * Jedro je okrogla Gaussova maska – vrednosti so visoke v sredini
 * obroča.
 */
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
        // Normalizacija jedra, da je vsota vseh elementov 1.0
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

    /*
     * OPTIMIZACIJA 2: Dvojno bufferiranje (Prostorska struktura)
     *
     * Da preprečimo prepisovanje podatkov, ki jih še potrebujemo za
     * izračun sosedov, rezerviramo dve ločeni matriki vnaprej:
     * 'world' (za branje trenutnega stanja) in 'next_world' (za zapis
     * novega stanja). To deluje v tesnem paru z Optimizacijo 7.
     */
    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));
    float *next_world = (float *)calloc(rows * cols, sizeof(float));

    w = generate_kernel(w, kernel_size);

    /*
     * Inicializacija orbiumov.
     * Funkcija place_orbium deluje z double natančnostjo, zato
     * najprej napolnimo začasno double matriko, nato vrednosti
     * prenesemo v našo float matriko world.
     */
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
        /*
         * OPTIMIZACIJA 3: Paralelizacija z OpenMP
         *
         * Vsaka vrstica je popolnoma neodvisna (beremo iz 'world',
         * pišemo v 'next_world'), zato zanko varno razdelimo med več
         * procesorskih niti. Ukaz schedule(static) enakomerno
         * in vnaprej razdeli vrstice.
         */
#pragma omp parallel for schedule(static)
        for (unsigned int i = 0; i < rows; i++)
        {
            /*
             * OPTIMIZACIJA 4: Vnaprejšnji izračun indeksov vrstic
             *
             * Indekse sosednjih vrstic (z upoštevanjem periodičnih robov)
             * izračunamo samo enkrat na začetku vrstice, namesto da bi jih
             * računali znova za vsak posamezen piksel v notranji zanki.
             */
            int row_indices[128];
            for (int ki = 0; ki < (int)kernel_size; ki++)
            {
                int ii = (int)i - r_k + ki;
                // Skrb za periodične robove mreže (torus)
                if (ii < 0)
                    ii += (int)rows;
                else if (ii >= (int)rows)
                    ii -= (int)rows;
                row_indices[ki] = ii * (int)rows;
            }

            for (unsigned int j = 0; j < cols; j++)
            {
                float sum = 0;
                /*
                 * Konvolucija z zrcaljenjem jedra (kernel flipping).
                 * Indeksa ki in kj gresta od kernel_size-1 do 0 (obratno),
                 * kar strogo ustreza matematični definiciji konvolucije.
                 */
                for (int ki = (int)kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    int row_offset = row_indices[kri];
                    for (int kj = (int)kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        /*
                         * OPTIMIZACIJA 5: Zamenjava operatorja % z if stavkom
                         *
                         * Modulo operator (%) je na procesorju počasna operacija
                         * (zahteva strojno deljenje). Ker indeks z vsakim korakom
                         * prestopi rob največ za 1, je preprost preverjalni 'if'
                         * stavek (seštevanje/odštevanje) bistveno hitrejši.
                         */
                        int jj = (int)j - r_k + kcj;
                        if (jj < 0)
                            jj += (int)cols;
                        else if (jj >= (int)cols)
                            jj -= (int)cols;

                        sum += w[ki * kernel_size + kj] * world[row_offset + jj];
                    }
                }

                /*
                 * OPTIMIZACIJA 6: Zlivanje zank (Loop fusion) in obvod
                 *
                 * Izračun rasti in zapis v 'next_world' opravimo neposredno
                 * po konvoluciji znotraj iste zanke, namesto v dveh ločenih prehodih.
                 * S tem drastično zmanjšamo število dostopov do pomnilnika (RAM-a).
                 *
                 * Če je okolica prazna (sum < epsilon), prihranimo
                 * kompleksen Gaussov izračun rasti in celico le linearno zmanjšamo.
                 */
                float res;
                if (sum < 1e-6f)
                {
                    res = world[i * (int)rows + j] - dt;
                }
                else
                {
                    res = world[i * (int)rows + j] + dt * growth_lenia(sum);
                }

                // Vrednost celice omejimo na veljaven interval [0, 1]
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

        /*
         * OPTIMIZACIJA 7: Zamenjava kazalcev
         *
         * Namesto fizičnega prekopiranja na tisoče elementov iz 'next_world'
         * nazaj v 'world', zgolj zamenjamo naslova, kamor kazalca kažeta.
         */
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