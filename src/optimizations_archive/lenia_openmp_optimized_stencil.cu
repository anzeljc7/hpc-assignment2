#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// CUDA knjižnice
#include <cuda_runtime.h>
#include <cuda.h>

// #define GENERATE_GIF

// KONFIGURACIJA OBLIKE BLOKA NITI (Thread Block Configuration)
// Spremeni to vrednost pred prevajanjem (kompilacijo) za izvedbo meritev.
//
// 0 = KVADRATI (32 x 32) -> Pričakovano najhitreje.
// 1 = VODORAVNI PASOVI (256 x 4)
// 2 = NAVPIČNI PASOVI (1024 x 1)
#define BLOCK_CONFIG 0

// 1. CUDA DEVICE FUNKCIJE (Matematika, ki se izvaja neposredno na GPU)
//    __device__: Funkcije so vidne in klicane izključno znotraj grafične kartice.
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

// 2. CUDA KERNEL (Osnovni pristop z globalnim pomnilnikom)
//    __global__: Kliče CPU, izvaja GPU.
//    V tej različici VSAKA nit bere VSE sosede neposredno iz glavnega pomnilnika.
__global__ void evolve_lenia_kernel(const float *world, float *next_world, const float *w,
                                    int rows, int cols, int kernel_size, float dt)
{
    // Vsaka nit izračuna svoj absolutni globalni indeks v 2D mreži
    int j = blockIdx.x * blockDim.x + threadIdx.x; // x koordinata (stolpec)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y koordinata (vrstica)

    // Varnostno preverjanje: niti, ki padejo izven dejanske slike, ne počnejo ničesar
    if (i < rows && j < cols)
    {
        float sum = 0.0f;
        int r_k = kernel_size / 2;

        for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
        {
            // Izračun vrstice in skrb za periodični rob (Wrap-around)
            // Uporabljamo hitrejše seštevanje/odštevanje namesto počasnega deljenja (%)
            int ii = i - r_k + kri;
            if (ii < 0)
                ii += rows;
            else if (ii >= rows)
                ii -= rows;

            for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
            {
                // Izračun stolpca in skrb za periodični rob
                int jj = j - r_k + kcj;
                if (jj < 0)
                    jj += cols;
                else if (jj >= cols)
                    jj -= cols;

                // Neposredno branje iz počasnejšega globalnega pomnilnika (world)
                // Uporabljamo 1D indeksiranje (vrstica * širina + stolpec) za 2D matriko
                sum += w[ki * kernel_size + kj] * world[ii * cols + jj];
            }
        }

        // Zlivanje zank (Loop Fusion): Izračun rasti in zapis v enem koraku
        float res;
        if (sum < 1e-6f) // Hitri preskok za mrtve celice
        {
            res = world[i * cols + j] - dt;
        }
        else
        {
            res = world[i * cols + j] + dt * growth_lenia_dev(sum);
        }

        // Omejitev vrednosti celice strogo na interval [0.0, 1.0]
        if (res > 1.0f)
            res = 1.0f;
        else if (res < 0.0f)
            res = 0.0f;

        next_world[i * cols + j] = res;
    }
}

// -------------------------------------------------------------------------
// 3. GENERIRANJE JEDRA NA CPU (Priprava začetnih podatkov)
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
    // Normalizacija
    for (unsigned int i = 0; i < size * size; i++)
        K[i] /= sum;
    return K;
}

// -------------------------------------------------------------------------
// 4. GLAVNA FUNKCIJA (Host Code - Izvaja centralni procesor)
// -------------------------------------------------------------------------
float *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps,
                    const float dt, const unsigned int kernel_size,
                    const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    // Priprava začetnega stanja v sistemskem pomnilniku (RAM)
    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));

    generate_kernel(w, kernel_size);

    // Inicializacija bitij (place_orbium uporablja double, zato prenašamo v float)
    double *d_world_cpu = (double *)calloc(rows * cols, sizeof(double));
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        d_world_cpu = place_orbium(d_world_cpu, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }
    for (unsigned int i = 0; i < rows * cols; i++)
        world[i] = (float)d_world_cpu[i];
    free(d_world_cpu);

    // =========================================================================
    // CUDA ALOKACIJA IN PRENOS (CPU -> GPU)
    // =========================================================================
    float *d_w, *d_world, *d_next_world;
    size_t world_bytes = rows * cols * sizeof(float);
    size_t w_bytes = kernel_size * kernel_size * sizeof(float);

    cudaMalloc((void **)&d_w, w_bytes);
    cudaMalloc((void **)&d_world, world_bytes);
    cudaMalloc((void **)&d_next_world, world_bytes);

    // Prenos iz RAM-a v VRAM
    cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_world, world, world_bytes, cudaMemcpyHostToDevice);

    // =========================================================================
    // NASTAVITEV MREŽE IN OBLIKE BLOKOV (Glede na izbrano konfiguracijo)
    // =========================================================================
    dim3 threadsPerBlock;
    if (BLOCK_CONFIG == 0)
    {
        threadsPerBlock = dim3(32, 32); // Kvadrat, 1024 niti na blok
    }
    else if (BLOCK_CONFIG == 1)
    {
        threadsPerBlock = dim3(256, 4); // Vodoravni pas, 256 niti na blok
    }
    else
    {
        threadsPerBlock = dim3(1024, 1); // Navpični pas, 256 niti na blok
    }

    // Izračun števila blokov, ki jih potrebujemo, da pokrijemo celotno sliko
    // (Dodamo threadsPerBlock - 1 za varno zaokroževanje navzgor)
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // =========================================================================
    // GLAVNA ZANKA SIMULACIJE (Hitro izvajanje na GPU)
    // =========================================================================
    for (unsigned int step = 0; step < steps; step++)
    {
        // Zagon CUDA kernela na tisočih jedrih hkrati
        evolve_lenia_kernel<<<numBlocks, threadsPerBlock>>>(d_world, d_next_world, d_w, rows, cols, kernel_size, dt);

        // O(1) zamenjava kazalcev na GPU (Pointer swap), da se izognemo kopiranju
        float *d_temp = d_world;
        d_world = d_next_world;
        d_next_world = d_temp;

#ifdef GENERATE_GIF
        // OPOZORILO: Ta prenos drastično zniža performanse celotnega programa!
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

    // Končni prenos rezultatov z GPU nazaj na CPU
    cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);

    // Sprostitev pomnilnika na grafični kartici
    cudaFree(d_w);
    cudaFree(d_world);
    cudaFree(d_next_world);
    free(w);

    return world;
}