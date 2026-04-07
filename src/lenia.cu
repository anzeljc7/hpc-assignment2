#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// CUDA knjižnice
#include <cuda_runtime.h>
#include <cuda.h>

// Makro za vklop generiranja animacije
// POMEMBNO: Za meritve časa to obvezno ZAKOMENTIRAJ, saj prenašanje na CPE v vsakem koraku ubije GPU hitrost!
// #define GENERATE_GIF

// ====================================================================================
// KONFIGURACIJA OBLIKE BLOKA (Spremeni pred prevajanjem za meritve za poročilo)
// 0 = KVADRATI (32 x 32) -> Pričakovano najhitreje zaradi boljšega predpomnilnika
// 1 = VODORAVNI PASOVI (256 x 1)
// 2 = NAVPIČNI PASOVI (1 x 256)
// ====================================================================================
#define BLOCK_CONFIG 0

// -------------------------------------------------------------------------
// 1. CUDA DEVICE FUNKCIJE (Matematika, ki se izvaja neposredno na GPU)
// -------------------------------------------------------------------------
__device__ float gauss_dev(float x, float mu, float sigma)
{
    float diff = (x - mu) / sigma;
    return expf(-0.5f * (diff * diff));
}

__device__ float growth_lenia_dev(float u)
{
    float mu = 0.15f;
    float sigma = 0.015f;
    return -1.0f + 2.0f * gauss_dev(u, mu, sigma);
}

// -------------------------------------------------------------------------
// 2. CUDA KERNEL (Osnovni Stencil vzorec v globalnem pomnilniku)
// -------------------------------------------------------------------------
__global__ void evolve_lenia_kernel(const float *world, float *next_world, const float *w,
                                    int rows, int cols, int kernel_size, float dt)
{
    // Vsaka nit (thread) izračuna svoj globalni indeks (eno celico v mreži)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // x koordinata (stolpec)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y koordinata (vrstica)

    // Preverimo, da nismo izven meja mreže
    if (i < rows && j < cols)
    {
        float sum = 0.0f;
        int r_k = kernel_size / 2;

        // Toroidalna konvolucija (Stencil)
        for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
        {
            // Izračun vrstice in toroidalni popravek (brez modula % za večjo hitrost)
            int ii = i - r_k + kri;
            if (ii < 0)
                ii += rows;
            else if (ii >= rows)
                ii -= rows;

            for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
            {
                // Izračun stolpca in toroidalni popravek
                int jj = j - r_k + kcj;
                if (jj < 0)
                    jj += cols;
                else if (jj >= cols)
                    jj -= cols;

                // Indeksiranje i * cols + j zagotavlja pravilno delovanje za vse dimenzije
                sum += w[ki * kernel_size + kj] * world[ii * cols + jj];
            }
        }

        // Združitev: Takojšnja posodobitev celice (Loop Fusion)
        float res;
        if (sum < 1e-6f)
        {
            res = world[i * cols + j] - dt;
        }
        else
        {
            res = world[i * cols + j] + dt * growth_lenia_dev(sum);
        }

        // Omejitev na interval [0, 1]
        if (res > 1.0f)
            res = 1.0f;
        else if (res < 0.0f)
            res = 0.0f;

        next_world[i * cols + j] = res;
    }
}

// -------------------------------------------------------------------------
// 3. GENERIRANJE JEDRA NA CPE
// -------------------------------------------------------------------------
float *generate_kernel(float *K, const unsigned int size)
{
    float mu = 0.5f;
    float sigma = 0.15f;
    int r = (int)size / 2;
    float sum = 0;
    for (int y = -r; y < r; y++)
    {
        for (int x = -r; x < r; x++)
        {
            float distance = sqrtf((float)((1 + x) * (1 + x) + (1 + y) * (1 + y))) / r;
            if (distance > 1.0f)
                K[(y + r) * size + x + r] = 0.0f;
            else
            {
                float diff = (distance - mu) / sigma;
                K[(y + r) * size + x + r] = expf(-0.5f * (diff * diff));
            }
            sum += K[(y + r) * size + x + r];
        }
    }
    for (unsigned int i = 0; i < size * size; i++)
        K[i] /= sum;
    return K;
}

// -------------------------------------------------------------------------
// 4. GLAVNA FUNKCIJA (Host Code)
// -------------------------------------------------------------------------
float *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const float dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    // Priprava začetnega stanja na CPU
    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));

    generate_kernel(w, kernel_size);

    // Inicializacija bitij (place_orbium uporablja double, zato vmesna tabela)
    double *d_world_cpu = (double *)calloc(rows * cols, sizeof(double));
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        d_world_cpu = place_orbium(d_world_cpu, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }
    for (unsigned int i = 0; i < rows * cols; i++)
        world[i] = (float)d_world_cpu[i];
    free(d_world_cpu);

    // =========================================================================
    // CUDA ALOKACIJA IN PRENOS (Host -> Device)
    // =========================================================================
    float *d_w, *d_world, *d_next_world;
    size_t world_bytes = rows * cols * sizeof(float);
    size_t w_bytes = kernel_size * kernel_size * sizeof(float);

    cudaMalloc((void **)&d_w, w_bytes);
    cudaMalloc((void **)&d_world, world_bytes);
    cudaMalloc((void **)&d_next_world, world_bytes);

    cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_world, world, world_bytes, cudaMemcpyHostToDevice);

    // =========================================================================
    // NASTAVITEV OBLIKE BLOKOV (Glede na konfiguracijo)
    // =========================================================================
    dim3 threadsPerBlock;
    if (BLOCK_CONFIG == 0)
    {
        threadsPerBlock = dim3(32, 32); // Kvadrat, 1024 niti
    }
    else if (BLOCK_CONFIG == 1)
    {
        threadsPerBlock = dim3(256, 1); // Vodoravni pas, 256 niti
    }
    else
    {
        threadsPerBlock = dim3(1, 256); // Navpični pas, 256 niti
    }

    // Izračun mreže (Grid)
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // =========================================================================
    // GLAVNA ZANKA SIMULACIJE (Na GPE)
    // =========================================================================
    for (unsigned int step = 0; step < steps; step++)
    {
        // Zagon CUDA kernela
        evolve_lenia_kernel<<<numBlocks, threadsPerBlock>>>(d_world, d_next_world, d_w, rows, cols, kernel_size, dt);

        // Zamenjava kazalcev na napravi (brez prenašanja podatkov)
        float *d_temp = d_world;
        d_world = d_next_world;
        d_next_world = d_temp;

#ifdef GENERATE_GIF
        // Če delamo GIF, moramo kopirati nazaj na CPU. TO UPOČASNI SIMULACIJO!
        cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                gif->frame[i * cols + j] = (uint8_t)(world[i * cols + j] * 255.0f);
            }
        }
        ge_add_frame(gif, 5);
#endif
    }

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    // Prenos končnega rezultata nazaj na CPE
    cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);

    // Čiščenje GPU pomnilnika
    cudaFree(d_w);
    cudaFree(d_world);
    cudaFree(d_next_world);
    free(w);

    return world;
}