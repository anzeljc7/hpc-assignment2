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
// Za meritve hitrosti mora biti zakomentirano, ker cudaMemcpy vsak korak
// drastično upočasni simulacijo (prenos GPU -> CPU čez PCIe vodilo)
// #define GENERATE_GIF

// ====================================================================================
// KONFIGURACIJA OBLIKE BLOKA
// 0 = KVADRATI (32 x 32) -> Optimalno za vse velikosti (256, 512, 1024, 2048, 4096)
//     ker so vse potence dvojke deljive z 32 -> nič idle niti
//     + najmanjše razmerje obseg/površina -> najmanj redundantnih branj
//     + memory coalescing (warp bere zaporedne naslove)
// 1 = VODORAVNI PASOVI (256 x 1) -> slabše (~9x več redundantnih branj)
// 2 = NAVPIČNI PASOVI (1 x 256)  -> najslabše (+ brez memory coalescinga)
// ====================================================================================
#define BLOCK_CONFIG 0

// -------------------------------------------------------------------------
// 1. CUDA DEVICE FUNKCIJE (Matematika, ki se izvaja neposredno na GPU)
//    __device__ pomeni: kliče jih GPU nit, CPU jih ne more klicati
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
// 2. CUDA KERNEL Z SHARED MEMORY
//    __global__ pomeni: sproži ga CPU, izvaja pa GPU
//    Vsaka nit izračuna točno eno celico mreže
//
//    Optimizacija z Shared Memory (po vzoru heat.cu):
//    - Vsak blok niti naloži svoj "tile" podatkov v shared memory
//    - Tile vključuje dejanske celice bloka + halo rob debeline r_k
//    - Shared memory je ~100x hitrejša od globalnega pomnilnika (VRAM)
//    - Brez shared memory bi vsaka nit brala iste sosede neodvisno
//      iz počasnega globalnega pomnilnika
// -------------------------------------------------------------------------
__global__ void evolve_lenia_kernel(const float *world, float *next_world, const float *w,
                                    int rows, int cols, int kernel_size, float dt)
{
    // -----------------------------------------------------------------------
    // SHARED MEMORY — deklaracija tile-a (dinamična velikost, podana ob klicu)
    // Velikost: (blockDim.x + 2*r_k) * (blockDim.y + 2*r_k) floatov
    // Za 32x32 blok in kernel_size=27 (r_k=13):
    //   tile = (32+26) * (32+26) * 4B = 58 * 58 * 4B ≈ 13KB  (vejde v 48KB limit)
    // -----------------------------------------------------------------------
    extern __shared__ float tile[];

    int r_k = kernel_size / 2;

    // Dimenzije tile-a (dejanski blok + halo na obeh straneh)
    int tileWidth  = blockDim.x + 2 * r_k;
    int tileHeight = blockDim.y + 2 * r_k;

    // Globalni indeksi te niti (katera celica v mreži)
    int gj = blockIdx.x * blockDim.x + threadIdx.x; // stolpec
    int gi = blockIdx.y * blockDim.y + threadIdx.y; // vrstica

    // Lokalni indeksi v tile-u (z offsetom r_k za halo, kot heat.cu uporablja +1)
    int tx = threadIdx.x + r_k;
    int ty = threadIdx.y + r_k;

    // -----------------------------------------------------------------------
    // NALAGANJE TILE + HALO IZ GLOBALNEGA POMNILNIKA V SHARED MEMORY
    //
    // V heat.cu je bil halo debeline 1, zato so zadoščali if stavki za 4 robove.
    // V Lenia je halo debeline r_k=13, zato potrebujemo zanko — vsaka nit
    // naloži več elementov (tile je bistveno večji od samega bloka).
    //
    // Vsaka nit v zanki pokrije del tile-a s korakom blockDim.x/y
    // -----------------------------------------------------------------------
    for (int dy = threadIdx.y; dy < tileHeight; dy += blockDim.y)
    {
        for (int dx = threadIdx.x; dx < tileWidth; dx += blockDim.x)
        {
            // Globalni koordinati tega elementa tile-a
            int src_i = (int)(blockIdx.y * blockDim.y) + dy - r_k;
            int src_j = (int)(blockIdx.x * blockDim.x) + dx - r_k;

            // Toroidalni wrap — brez modula % (if/else je hitrejši)
            if (src_i < 0)           src_i += rows;
            else if (src_i >= rows)  src_i -= rows;
            if (src_j < 0)           src_j += cols;
            else if (src_j >= cols)  src_j -= cols;

            // Naložimo vrednost iz globalnega pomnilnika v shared memory
            tile[dy * tileWidth + dx] = world[src_i * cols + src_j];
        }
    }

    // -----------------------------------------------------------------------
    // SINHRONIZACIJA — počakamo da VSE niti v bloku dokončajo nalaganje
    // Brez tega bi nekatere niti začele računati preden so sosednje niti
    // naložile halo (race condition)
    // Enako kot v heat.cu: __syncthreads()
    // -----------------------------------------------------------------------
    __syncthreads();

    // -----------------------------------------------------------------------
    // KONVOLUCIJA — beremo iz TILE (shared memory) namesto world (globalni)
    //
    // tx, ty že vključujeta r_k offset, zato:
    //   tile[ty + kri - r_k][tx + kcj - r_k] = tile[threadIdx.y + kri][threadIdx.x + kcj]
    // -----------------------------------------------------------------------
    if (gi < rows && gj < cols)
    {
        float sum = 0.0f;

        // Zrcaljena konvolucija (kernel flip)
        for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
        {
            for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
            {
                // Koordinate v tile-u — ty/tx že vsebujeta r_k offset
                int tile_i = ty - r_k + kri;  // = threadIdx.y + kri
                int tile_j = tx - r_k + kcj;  // = threadIdx.x + kcj

                // Beremo iz shared memory (hitro!) namesto iz globalnega (počasno)
                sum += w[ki * kernel_size + kj] * tile[tile_i * tileWidth + tile_j];
            }
        }

        // Loop Fusion: growth + clamp + zapis v enem prehodu

        // Optimizacija: preskočimo Gauss izračun pri mrtvih celicah (sum ≈ 0)
        float res;
        if (sum < 1e-6f)
        {
            res = world[gi * cols + gj] - dt;
        }
        else
        {
            res = world[gi * cols + gj] + dt * growth_lenia_dev(sum);
        }

        // Clamp na [0, 1] z if/else namesto fminf/fmaxf
        if (res > 1.0f)       res = 1.0f;
        else if (res < 0.0f)  res = 0.0f;

        next_world[gi * cols + gj] = res;
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
    // Normalizacija jedra
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

    // =========================================================================
    // PRIPRAVA ZAČETNEGA STANJA NA CPU
    // =========================================================================
    float *w     = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));

    generate_kernel(w, kernel_size);

    // Inicializacija orbiumov (place_orbium uporablja double, zato vmesna tabela)
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
    // CUDA ALOKACIJA IN PRENOS — HOST -> DEVICE
    // Prenosi se zgodijo samo ENKRAT pred zanko (ne znotraj nje)
    // ①  w      (kernel jedro)    CPU -> GPU
    // ②  world  (začetno stanje)  CPU -> GPU
    // =========================================================================
    float *d_w, *d_world, *d_next_world;
    size_t world_bytes = rows * cols * sizeof(float);
    size_t w_bytes     = kernel_size * kernel_size * sizeof(float);

    cudaMalloc((void **)&d_w,          w_bytes);
    cudaMalloc((void **)&d_world,      world_bytes);
    cudaMalloc((void **)&d_next_world, world_bytes);

    cudaMemcpy(d_w,     w,     w_bytes,     cudaMemcpyHostToDevice);  // ①
    cudaMemcpy(d_world, world, world_bytes, cudaMemcpyHostToDevice);  // ②

    // =========================================================================
    // NASTAVITEV OBLIKE BLOKOV
    // Za vse naše velikosti (256, 512, 1024, 2048, 4096) je 32x32 optimalen:
    // - So potence dvojke → deljive z 32 → 0 idle niti
    // - Kvadrat minimizira obseg/površina razmerje → ~3.28x redundantnih branj
    //   (vs ~29.7x pri pasovih)
    // - Memory coalescing: warp (32 niti) bere zaporedne naslove
    // =========================================================================
    dim3 threadsPerBlock;
    if (BLOCK_CONFIG == 0)
        threadsPerBlock = dim3(32, 32);   // Kvadrat — optimalno
    else if (BLOCK_CONFIG == 1)
        threadsPerBlock = dim3(256, 1);   // Vodoravni pas
    else
        threadsPerBlock = dim3(1, 256);   // Navpični pas

    // Izračun mreže blokov (zaokrožimo navzgor za robne primere)
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // =========================================================================
    // IZRAČUN VELIKOSTI SHARED MEMORY (kot localMemSize v heat.cu)
    // tile = (blockDim.x + 2*r_k) * (blockDim.y + 2*r_k) * sizeof(float)
    // Za 32x32 blok, kernel_size=27, r_k=13:
    //   tile = 58 * 58 * 4B = 13.456B ≈ 13KB  (limit je 48KB na blok)
    // =========================================================================
    int r_k = kernel_size / 2;
    int localMemSize = (threadsPerBlock.x + 2 * r_k) *
                       (threadsPerBlock.y + 2 * r_k) * sizeof(float);

    // =========================================================================
    // GLAVNA ZANKA SIMULACIJE — teče v celoti na GPU
    // Nobenih prenosov CPU<->GPU znotraj zanke (razen opcijskega GIF-a)
    // =========================================================================
    for (unsigned int step = 0; step < steps; step++)
    {
        // Zagon kernela z dodatnim parametrom za shared memory
        // (kot v heat.cu: heatStep<<<grid, block, localMemSize>>>)
        evolve_lenia_kernel<<<numBlocks, threadsPerBlock, localMemSize>>>(
            d_world, d_next_world, d_w, rows, cols, kernel_size, dt
        );

        // Pointer swap — zamenjamo naslova brez kopiranja podatkov
        float *d_temp  = d_world;
        d_world        = d_next_world;
        d_next_world   = d_temp;

#ifdef GENERATE_GIF
        // POZOR: Ta prenos se ne šteje v čas meritev, a vseeno upočasni simulacijo
        // GPU -> CPU prenos čez PCIe vodilo vsak korak
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

    // ③ Prenos končnega rezultata — DEVICE -> HOST (enkrat, po zanki)
    cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);  // ③

    // Čiščenje GPU pomnilnika
    cudaFree(d_w);
    cudaFree(d_world);
    cudaFree(d_next_world);
    free(w);

    return world;
}
