#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

#include <cuda_runtime.h>
#include <cuda.h>

// #define GENERATE_GIF

// KONFIGURACIJA OBLIKE BLOKA (NITI)
// 0 = KVADRATI (32 x 32) -> Optimalno za dimenzije
// 1 = VODORAVNI PASOVI (256 x 4)
// 2 = NAVPIČNI PASOVI (1024 x 1)
#define BLOCK_CONFIG 1

// 1. CUDA DEVICE FUNKCIJE (Matematika, ki se izvaja neposredno na GPU)
//    __device__: Te funkcije lahko kliče samo GPU. CPU do njih nima dostopa.
__device__ float gauss_dev(float x, float mu, float sigma)
{
    float diff = (x - mu) / sigma;
    return expf(-0.5f * (diff * diff)); // expf() je optimizirana različica za float
}

__device__ float growth_lenia_dev(float u)
{
    float mu = 0.15f;
    float sigma = 0.015f;
    return -1.0f + 2.0f * gauss_dev(u, mu, sigma);
}

// 2. CUDA KERNEL Z DELJENIM POMNILNIKOM (SHARED MEMORY)
//    __global__: Funkcijo sproži CPU, izvaja pa se vzporedno na GPU.
//    Vsaka nit (thread) izračuna vrednost točno ene celice v mreži.
__global__ void evolve_lenia_kernel(const float *world, float *next_world, const float *w,
                                    int rows, int cols, int kernel_size, float dt)
{
    // -----------------------------------------------------------------------
    // DELJENI POMNILNIK (SHARED MEMORY) — Deklaracija bloka podatkov (tile)
    // Dinamična velikost je določena ob klicu kernela.
    // Vsak blok niti naloži svoj del mreže (tile) + potreben zunanji rob (halo/ghost cells).
    // Deljeni pomnilnik je na samem čipu (L1 cache nivo) in je ~100x hitrejši od VRAM-a.
    // -----------------------------------------------------------------------
    extern __shared__ float tile[];

    int r_k = kernel_size / 2;

    // Dimenzije tile-a (velikost bloka niti + debelina roba r_k na vseh 4 straneh)
    int tileWidth = blockDim.x + 2 * r_k;
    int tileHeight = blockDim.y + 2 * r_k;

    // Globalne koordinate te niti v celotni mreži
    int gj = blockIdx.x * blockDim.x + threadIdx.x; // Stolpec
    int gi = blockIdx.y * blockDim.y + threadIdx.y; // Vrstica

    // Lokalne koordinate te niti znotraj tile-a (premaknjene za debelino roba r_k)
    int tx = threadIdx.x + r_k;
    int ty = threadIdx.y + r_k;

    // -----------------------------------------------------------------------
    // SKUPINSKO NALAGANJE PODATKOV IZ GLOBALNEGA V DELJENI POMNILNIK
    // Ker je tile (blok + rob) večji od števila niti v bloku, mora vsaka nit
    // naložiti več kot le en element. S spodnjo zanko niti sodelujejo pri
    // prenosu celotnega območja.
    // -----------------------------------------------------------------------
    for (int dy = threadIdx.y; dy < tileHeight; dy += blockDim.y)
    {
        for (int dx = threadIdx.x; dx < tileWidth; dx += blockDim.x)
        {
            // Izračun globalnih koordinat za branje
            int src_i = (int)(blockIdx.y * blockDim.y) + dy - r_k;
            int src_j = (int)(blockIdx.x * blockDim.x) + dx - r_k;

            // Toroidalni svet (periodični robovi) - uporaba hitrih 'if' stavkov namesto '%'
            if (src_i < 0)
                src_i += rows;
            else if (src_i >= rows)
                src_i -= rows;
            if (src_j < 0)
                src_j += cols;
            else if (src_j >= cols)
                src_j -= cols;

            // Zapis v hitri deljeni pomnilnik
            tile[dy * tileWidth + dx] = world[src_i * cols + src_j];
        }
    }

    // -----------------------------------------------------------------------
    // SINHRONIZACIJA NITI
    // Nujno! Počakamo, da so VSE niti v bloku zaključile s prenosom podatkov.
    // Brez tega bi nekatere niti začele računati konvolucijo z neobstoječimi podatki.
    // -----------------------------------------------------------------------
    __syncthreads();

    // -----------------------------------------------------------------------
    // KONVOLUCIJA (Izračun na podlagi hitrega deljenega pomnilnika)
    // -----------------------------------------------------------------------
    if (gi < rows && gj < cols)
    {
        float sum = 0.0f;

        // Zrcaljena konvolucija po definiciji
        for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
        {
            for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
            {
                // Lokalne koordinate znotraj tile-a
                int tile_i = ty - r_k + kri;
                int tile_j = tx - r_k + kcj;

                // Branje je izjemno hitro, ker beremo iz spremenljivke 'tile'
                sum += w[ki * kernel_size + kj] * tile[tile_i * tileWidth + tile_j];
            }
        }

        // Hitri preskok za prazna območja (mrtve celice)
        float res;
        if (sum < 1e-6f)
        {
            res = world[gi * cols + gj] - dt;
        }
        else
        {
            res = world[gi * cols + gj] + dt * growth_lenia_dev(sum);
        }

        // Omejitev vrednosti (Clamp) med 0.0 in 1.0
        if (res > 1.0f)
            res = 1.0f;
        else if (res < 0.0f)
            res = 0.0f;

        next_world[gi * cols + gj] = res;
    }
}

// -------------------------------------------------------------------------
// 3. GENERIRANJE JEDRA NA CPE (Priprava na procesorju)
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
// 4. GLAVNA FUNKCIJA (Host Code - Vodi jo CPU)
// -------------------------------------------------------------------------
float *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps,
                    const float dt, const unsigned int kernel_size,
                    const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    // =========================================================================
    // PRIPRAVA ZAČETNEGA STANJA NA CPU (Host)
    // =========================================================================
    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));

    generate_kernel(w, kernel_size);

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
    // CUDA ALOKACIJA IN PRENOS (CPU -> GPU)
    // Izvede se samo enkrat pred zanko! Komunikacija čez PCIe vodilo je počasna.
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
    // NASTAVITEV MREŽE IN BLOKOV ZA GPU
    // =========================================================================
    dim3 threadsPerBlock;
    if (BLOCK_CONFIG == 0)
        threadsPerBlock = dim3(32, 32); // Kvadrat — optimalno za grafične kartice
    else if (BLOCK_CONFIG == 1)
        threadsPerBlock = dim3(256, 4); // Vodoravni pas
    else
        threadsPerBlock = dim3(1024, 1); // Navpični pas

    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // =========================================================================
    // IZRAČUN VELIKOSTI DELJENEGA POMNILNIKA (Shared Memory)
    // Formula: (širina bloka + 2*rob) * (višina bloka + 2*rob) * velikost_tipa
    // =========================================================================
    int r_k = kernel_size / 2;
    int localMemSize = (threadsPerBlock.x + 2 * r_k) *
                       (threadsPerBlock.y + 2 * r_k) * sizeof(float);

    // =========================================================================
    // GLAVNA ZANKA SIMULACIJE (Izvaja se izključno na GPU)
    // =========================================================================
    for (unsigned int step = 0; step < steps; step++)
    {
        // Klic kernela: določimo mrežo blokov, velikost bloka in velikost deljenega pomnilnika
        evolve_lenia_kernel<<<numBlocks, threadsPerBlock, localMemSize>>>(
            d_world, d_next_world, d_w, rows, cols, kernel_size, dt);

        // Zamenjava kazalcev na GPU (O(1) operacija)
        float *d_temp = d_world;
        d_world = d_next_world;
        d_next_world = d_temp;

#ifdef GENERATE_GIF
        // Prenos stanja z GPU na CPU za shranjevanje okvirja
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

    // Prenos končnega rezultata nazaj na CPU
    cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);

    // Čiščenje GPU pomnilnika
    cudaFree(d_w);
    cudaFree(d_world);
    cudaFree(d_next_world);
    free(w);

    return world;
}