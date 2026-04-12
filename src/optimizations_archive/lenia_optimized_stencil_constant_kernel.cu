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
// #define GENERATE_GIF

// ====================================================================================
// KONFIGURACIJA OBLIKE BLOKA (Spremeni pred prevajanjem za meritve za poročilo)
// 0 = KVADRATI (32 x 32) -> Optimalno za vse velikosti (256, 512, 1024, 2048, 4096)
// 1 = VODORAVNI PASOVI (256 x 1)
// 2 = NAVPIČNI PASOVI (1 x 256)
// ====================================================================================
#define BLOCK_CONFIG 0

// =========================================================================
// __CONSTANT__ MEMORY ZA KONVOLUCIJSKO JEDRO w
//
// w se med simulacijo NIKOLI ne spremeni → idealen kandidat za __constant__
//
// Razlika od globalnega pomnilnika (d_w):
//   Globalni:    vsaka nit bere w[i] posebej → 729 × 1024 = 746.496 dostopov/blok
//   __constant__: GPU vidi da vse niti berejo isti w[i] v istem koraku zanke
//                 → BROADCAST: 1 dostop za vse niti hkrati → ~729 dostopov/blok
//
// Omejitev: max 64KB → 27×27×4B = 2.916B ✓ (daleč pod limitom)
//
// Mora biti globalna spremenljivka (izven vseh funkcij) — to je zahteva CUDA
// =========================================================================
#define MAX_KERNEL_ELEMS (27 * 27)
__constant__ float d_w_const[MAX_KERNEL_ELEMS];

// -------------------------------------------------------------------------
// 1. CUDA DEVICE FUNKCIJE (Matematika, ki se izvaja neposredno na GPU)
// -------------------------------------------------------------------------
__device__ float gauss_dev(float x, float mu, float sigma)
{
    float diff = (x - mu) / sigma;
    return expf(-0.5f * (diff * diff)); // expf() je hitrejša float različica exp()
}

__device__ float growth_lenia_dev(float u)
{
    float mu = 0.15f;
    float sigma = 0.015f;
    return -1.0f + 2.0f * gauss_dev(u, mu, sigma);
}

// -------------------------------------------------------------------------
// 2. CUDA KERNEL
//    Sprememba glede na original:
//    - parameter "const float *w" je ODSTRANJEN
//    - namesto tega beremo iz d_w_const (broadcast cache)
// -------------------------------------------------------------------------
__global__ void evolve_lenia_kernel(const float *world, float *next_world,
                                    int rows, int cols, int kernel_size, float dt)
{
    // Vsaka nit izračuna svoj globalni indeks (eno celico v mreži)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // x koordinata (stolpec)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y koordinata (vrstica)

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

                // Beremo iz __constant__ memory namesto globalnega pomnilnika
                // Vse niti v warpu berejo isti d_w_const[...] v istem koraku
                // → GPU to pretvori v en sam broadcast dostop za cel warp
                sum += d_w_const[ki * kernel_size + kj] * world[ii * cols + jj];
            }
        }

        // Loop Fusion: growth + clamp v enem prehodu
        float res;
        if (sum < 1e-6f)
        {
            res = world[i * cols + j] - dt;
        }
        else
        {
            res = world[i * cols + j] + dt * growth_lenia_dev(sum);
        }

        // Clamp na [0, 1]
        if (res > 1.0f)
            res = 1.0f;
        else if (res < 0.0f)
            res = 0.0f;

        next_world[i * cols + j] = res;
    }
}

// -------------------------------------------------------------------------
// 3. GENERIRANJE JEDRA NA CPE (nespremenjeno)
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
float *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps,
                    const float dt, const unsigned int kernel_size,
                    const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    // Priprava začetnega stanja na CPU
    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));

    generate_kernel(w, kernel_size);

    // Inicializacija orbiumov
    double *d_world_cpu = (double *)calloc(rows * cols, sizeof(double));
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        d_world_cpu = place_orbium(d_world_cpu, rows, cols,
                                   orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }
    for (unsigned int i = 0; i < rows * cols; i++)
        world[i] = (float)d_world_cpu[i];
    free(d_world_cpu);

    // =========================================================================
    // PRENOS JEDRA w V __CONSTANT__ MEMORY
    //
    // Namesto cudaMalloc + cudaMemcpy uporabimo cudaMemcpyToSymbol
    // To direktno napiše v constant cache — ni potrebe po cudaMalloc za w
    //
    // Original:
    //   cudaMalloc(&d_w, w_bytes);
    //   cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice);
    //
    // Novo:
    //   cudaMemcpyToSymbol(d_w_const, w, w_bytes);
    // =========================================================================
    size_t w_bytes = kernel_size * kernel_size * sizeof(float);
    cudaMemcpyToSymbol(d_w_const, w, w_bytes); // CPU → __constant__ memory

    // =========================================================================
    // CUDA ALOKACIJA IN PRENOS ZA world (Host -> Device)
    // Prenosi se zgodijo samo ENKRAT pred zanko
    // =========================================================================
    float *d_world, *d_next_world;
    size_t world_bytes = rows * cols * sizeof(float);

    cudaMalloc((void **)&d_world, world_bytes);
    cudaMalloc((void **)&d_next_world, world_bytes);

    cudaMemcpy(d_world, world, world_bytes, cudaMemcpyHostToDevice);

    // =========================================================================
    // NASTAVITEV OBLIKE BLOKOV
    // =========================================================================
    dim3 threadsPerBlock;
    if (BLOCK_CONFIG == 0)
        threadsPerBlock = dim3(32, 32); // Kvadrat — optimalno
    else if (BLOCK_CONFIG == 1)
        threadsPerBlock = dim3(256, 1); // Vodoravni pas
    else
        threadsPerBlock = dim3(1, 256); // Navpični pas

    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // =========================================================================
    // GLAVNA ZANKA SIMULACIJE (Na GPE)
    // =========================================================================
    for (unsigned int step = 0; step < steps; step++)
    {
        // Kernel ne potrebuje več w parametra — bere direktno iz d_w_const
        evolve_lenia_kernel<<<numBlocks, threadsPerBlock>>>(
            d_world, d_next_world, rows, cols, kernel_size, dt);

        // Pointer swap — zamenjamo naslova brez kopiranja podatkov
        float *d_temp = d_world;
        d_world = d_next_world;
        d_next_world = d_temp;

#ifdef GENERATE_GIF
        cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);
        for (unsigned int i = 0; i < rows; i++)
            for (unsigned int j = 0; j < cols; j++)
                gif->frame[i * cols + j] = (uint8_t)(world[i * cols + j] * 255.0f);
        ge_add_frame(gif, 5);
#endif
    }

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    // Prenos končnega rezultata — DEVICE -> HOST (enkrat, po zanki)
    cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);

    // Čiščenje GPU pomnilnika (d_w_const ne rabimo cudaFree — ni na heap-u)
    cudaFree(d_world);
    cudaFree(d_next_world);
    free(w);

    return world;
}
