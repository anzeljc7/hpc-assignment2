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
// KONFIGURACIJA OBLIKE BLOKA
// 0 = KVADRATI (32 x 32) -> Optimalno za vse velikosti (256, 512, 1024, 2048, 4096)
// 1 = VODORAVNI PASOVI (256 x 1)
// 2 = NAVPIČNI PASOVI (1 x 256)
// ====================================================================================
#define BLOCK_CONFIG 0

// -------------------------------------------------------------------------
// 1. CUDA DEVICE FUNKCIJE
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
// 2. CUDA KERNEL Z SHARED MEMORY
//
//    Za multi-GPU: vsak GPU dobi svojo polovico vrstic + halo rob
//    Kernel se ne spremeni — dela na svojem delu, ne ve za drugi GPU
//    Parameter "local_rows" je število vrstic ki jih ta GPU obdeluje (brez haló)
//    Parameter "offset" je globalni indeks prve vrstice tega GPU-ja
// -------------------------------------------------------------------------
__global__ void evolve_lenia_kernel(const float *world, float *next_world, const float *w,
                                    int local_rows, int cols, int kernel_size, float dt,
                                    int offset, int total_rows)
{
    extern __shared__ float tile[];

    int r_k = kernel_size / 2;
    int tileWidth = blockDim.x + 2 * r_k;
    int tileHeight = blockDim.y + 2 * r_k;

    // Lokalni indeks znotraj tega GPU-jevega dela (vključuje halo)
    int gj = blockIdx.x * blockDim.x + threadIdx.x;
    int gi = blockIdx.y * blockDim.y + threadIdx.y; // lokalna vrstica (brez haló)

    int tx = threadIdx.x + r_k;
    int ty = threadIdx.y + r_k;

    // -----------------------------------------------------------------------
    // NALAGANJE TILE + HALO
    // world na tem GPU-ju vsebuje: [halo_zgoraj | dejanske vrstice | halo_spodaj]
    // Indeks 0 v world ustreza vrstici (gi - r_k) v globalnem world
    // -----------------------------------------------------------------------
    for (int dy = threadIdx.y; dy < tileHeight; dy += blockDim.y)
    {
        for (int dx = threadIdx.x; dx < tileWidth; dx += blockDim.x)
        {
            // Lokalni koordinati v GPU-jevem world buffru (ki vključuje halo)
            int src_i = gi + dy - threadIdx.y; // lokalna vrstica v buffru
            int src_j = (int)(blockIdx.x * blockDim.x) + dx - r_k;

            // src_i je že v lokalnem koordinatnem sistemu (halo je na začetku bufra)
            // Samo stolpce je treba toroidalno zaviti
            if (src_j < 0)
                src_j += cols;
            else if (src_j >= cols)
                src_j -= cols;

            tile[dy * tileWidth + dx] = world[src_i * cols + src_j];
        }
    }

    __syncthreads();

    // Računamo samo dejanske vrstice (brez haló)
    if (gi >= r_k && gi < local_rows + r_k && gj < cols)
    {
        float sum = 0.0f;

        for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
        {
            for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
            {
                int tile_i = ty - r_k + kri;
                int tile_j = tx - r_k + kcj;
                sum += w[ki * kernel_size + kj] * tile[tile_i * tileWidth + tile_j];
            }
        }

        float res;
        if (sum < 1e-6f)
            res = world[gi * cols + gj] - dt;
        else
            res = world[gi * cols + gj] + dt * growth_lenia_dev(sum);

        if (res > 1.0f)
            res = 1.0f;
        else if (res < 0.0f)
            res = 0.0f;

        // Zapisujemo v next_world na lokalnem indeksu (halo se ne posodablja)
        next_world[gi * cols + gj] = res;
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
// 4. GLAVNA FUNKCIJA (Host Code) — Multi-GPU
// -------------------------------------------------------------------------
float *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps,
                    const float dt, const unsigned int kernel_size,
                    const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
    // =========================================================================
    // PREVERJANJE ŠTEVILA GPU-JEV
    // =========================================================================
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 2)
    {
        // Fallback: če ni 2 GPU-jev, opozorimo in zaženemo na enem
        printf("OPOZORILO: Najden %d GPU (potrebna 2). Zaganjam na 1 GPU.\n", num_gpus);
        num_gpus = 1;
    }
    else
    {
        printf("Najdeni %d GPU-ji. Zaganjam na 2 GPU-jih.\n", num_gpus);
        num_gpus = 2; // Uporabljamo točno 2
    }

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    // =========================================================================
    // PRIPRAVA ZAČETNEGA STANJA NA CPU
    // =========================================================================
    float *w = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float *world = (float *)calloc(rows * cols, sizeof(float));

    generate_kernel(w, kernel_size);

    double *d_world_cpu = (double *)calloc(rows * cols, sizeof(double));
    for (unsigned int o = 0; o < num_orbiums; o++)
        d_world_cpu = place_orbium(d_world_cpu, rows, cols,
                                   orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    for (unsigned int i = 0; i < rows * cols; i++)
        world[i] = (float)d_world_cpu[i];
    free(d_world_cpu);

    // =========================================================================
    // RAZDELITEV DELA MED GPU-JA
    //
    // GPU 0: vrstice [0        .. half-1     ] — zgornja polovica
    // GPU 1: vrstice [half     .. rows-1     ] — spodnja polovica
    //
    // Vsak GPU dobi buffr z:
    //   [halo_zgoraj (r_k vrstic) | dejanske vrstice (half) | halo_spodaj (r_k vrstic)]
    //
    // Skupaj: (half + 2*r_k) vrstic na GPU
    // =========================================================================
    int r_k = kernel_size / 2;
    int half0 = rows / 2;     // GPU 0 dobi polovico
    int half1 = rows - half0; // GPU 1 dobi preostanek (za liha rows)

    size_t w_bytes = kernel_size * kernel_size * sizeof(float);
    size_t buf0_bytes = (half0 + 2 * r_k) * cols * sizeof(float); // z halo
    size_t buf1_bytes = (half1 + 2 * r_k) * cols * sizeof(float);

    // Pointers za oba GPU-ja
    float *d_w[2], *d_world_g[2], *d_next_world_g[2];

    // =========================================================================
    // VKLOP PEER ACCESS — direktni prenos med GPU-jema brez CPU
    // Preveri ali sta GPU-ja na istem PCIe switch-u
    // =========================================================================
    int canAccess01 = 0, canAccess10 = 0;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);

    if (canAccess01 && canAccess10)
    {
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        printf("Peer access omogočen — direktni prenos med GPU-jema.\n");
    }
    else
    {
        printf("Peer access NI možen — halo gre čez CPU RAM (počasneje).\n");
    }

    // =========================================================================
    // ALOKACIJA IN INICIALIZACIJA NA VSAKEM GPU
    // =========================================================================
    for (int g = 0; g < num_gpus; g++)
    {
        cudaSetDevice(g);

        int local_half = (g == 0) ? half0 : half1;
        size_t buf_bytes = (local_half + 2 * r_k) * cols * sizeof(float);

        // Alokacija jedra w in worda (z halo) na vsakem GPU
        cudaMalloc((void **)&d_w[g], w_bytes);
        cudaMalloc((void **)&d_world_g[g], buf_bytes);
        cudaMalloc((void **)&d_next_world_g[g], buf_bytes);

        // Prenos jedra w na ta GPU
        cudaMemcpy(d_w[g], w, w_bytes, cudaMemcpyHostToDevice);

        // Inicializacija bufra na 0
        cudaMemset(d_world_g[g], 0, buf_bytes);
        cudaMemset(d_next_world_g[g], 0, buf_bytes);

        // Prenos dejanskih vrstic (brez haló) — začnemo na offsetu r_k v buffru
        // GPU 0: world[0        .. half0-1    ] → d_world_g[0][r_k .. r_k+half0-1]
        // GPU 1: world[half0    .. rows-1     ] → d_world_g[1][r_k .. r_k+half1-1]
        int global_row_start = (g == 0) ? 0 : half0;
        cudaMemcpy(
            d_world_g[g] + r_k * cols,       // dest: za halo
            world + global_row_start * cols, // src: CPU world
            local_half * cols * sizeof(float),
            cudaMemcpyHostToDevice);
    }

    // =========================================================================
    // NASTAVITEV BLOKOV IN GRIDA
    // =========================================================================
    dim3 threadsPerBlock;
    if (BLOCK_CONFIG == 0)
        threadsPerBlock = dim3(32, 32);
    else if (BLOCK_CONFIG == 1)
        threadsPerBlock = dim3(256, 1);
    else
        threadsPerBlock = dim3(1, 256);

    // Grid za vsak GPU — po lokalnem številu vrstic (+ halo za pokritost)
    dim3 numBlocks0((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (half0 + 2 * r_k + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 numBlocks1((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (half1 + 2 * r_k + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int localMemSize = (threadsPerBlock.x + 2 * r_k) *
                       (threadsPerBlock.y + 2 * r_k) * sizeof(float);

    // =========================================================================
    // GLAVNA ZANKA SIMULACIJE
    // =========================================================================
    for (unsigned int step = 0; step < steps; step++)
    {
        // -----------------------------------------------------------------------
        // KORAK 1: IZMENJAVA HALO VRSTIC MED GPU-JEMA (cudaMemcpyPeer)
        //
        // Preden zaženemo kernel, morata oba GPU-ja imeti ažurne halo vrstice
        // od soseda iz prejšnjega koraka.
        //
        // GPU 0 potrebuje spodnji halo  = zadnjih r_k vrstic GPU 1
        // GPU 1 potrebuje zgornji halo  = prvih r_k vrstic GPU 0
        //
        // Toroidalno:
        //   GPU 0 zgornji halo = zadnjih r_k vrstic GPU 1 (wrap)
        //   GPU 1 spodnji halo = prvih r_k vrstic GPU 0 (wrap)
        // -----------------------------------------------------------------------
        size_t halo_bytes = r_k * cols * sizeof(float);

        // GPU 0 spodnji halo ← GPU 1 dejanske prve vrstice
        cudaMemcpyPeer(
            d_world_g[0] + (r_k + half0) * cols, // dest: spodnji halo GPU 0
            0,                                   // dest device
            d_world_g[1] + r_k * cols,           // src: prve dejanske vrstice GPU 1
            1,                                   // src device
            halo_bytes);

        // GPU 1 zgornji halo ← GPU 0 dejanske zadnje vrstice
        cudaMemcpyPeer(
            d_world_g[1], // dest: zgornji halo GPU 1 (offset 0)
            1,
            d_world_g[0] + half0 * cols, // src: zadnje dejanske vrstice GPU 0
            0,
            halo_bytes);

        // Toroidalni wrap:
        // GPU 0 zgornji halo ← GPU 1 zadnje dejanske vrstice
        cudaMemcpyPeer(
            d_world_g[0], // dest: zgornji halo GPU 0
            0,
            d_world_g[1] + half1 * cols, // src: zadnje vrstice GPU 1
            1,
            halo_bytes);

        // GPU 1 spodnji halo ← GPU 0 prve dejanske vrstice
        cudaMemcpyPeer(
            d_world_g[1] + (r_k + half1) * cols, // dest: spodnji halo GPU 1
            1,
            d_world_g[0] + r_k * cols, // src: prve vrstice GPU 0
            0,
            halo_bytes);

        // -----------------------------------------------------------------------
        // KORAK 2: ZAGON KERNELA NA VSAKEM GPU
        // -----------------------------------------------------------------------
        cudaSetDevice(0);
        evolve_lenia_kernel<<<numBlocks0, threadsPerBlock, localMemSize>>>(
            d_world_g[0], d_next_world_g[0], d_w[0],
            half0, cols, kernel_size, dt, 0, rows);

        cudaSetDevice(1);
        evolve_lenia_kernel<<<numBlocks1, threadsPerBlock, localMemSize>>>(
            d_world_g[1], d_next_world_g[1], d_w[1],
            half1, cols, kernel_size, dt, half0, rows);

        // Počakamo da oba GPU-ja dokončata
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        cudaSetDevice(1);
        cudaDeviceSynchronize();

        // -----------------------------------------------------------------------
        // KORAK 3: POINTER SWAP NA VSAKEM GPU
        // -----------------------------------------------------------------------
        for (int g = 0; g < num_gpus; g++)
        {
            float *d_temp = d_world_g[g];
            d_world_g[g] = d_next_world_g[g];
            d_next_world_g[g] = d_temp;
        }

#ifdef GENERATE_GIF
        // Prenos obeh polovic nazaj na CPU za GIF
        cudaSetDevice(0);
        cudaMemcpy(world,
                   d_world_g[0] + r_k * cols, // preskoči zgornji halo
                   half0 * cols * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaSetDevice(1);
        cudaMemcpy(world + half0 * cols,
                   d_world_g[1] + r_k * cols, // preskoči zgornji halo
                   half1 * cols * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (unsigned int i = 0; i < rows; i++)
            for (unsigned int j = 0; j < cols; j++)
                gif->frame[i * cols + j] = (uint8_t)(world[i * cols + j] * 255.0f);
        ge_add_frame(gif, 5);
#endif
    }

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif

    // =========================================================================
    // PRENOS KONČNEGA REZULTATA — oba GPU-ja → CPU
    // GPU 0: vrstice 0     .. half0-1
    // GPU 1: vrstice half0 .. rows-1
    // =========================================================================
    cudaSetDevice(0);
    cudaMemcpy(world,
               d_world_g[0] + r_k * cols, // preskoči zgornji halo
               half0 * cols * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaSetDevice(1);
    cudaMemcpy(world + half0 * cols,
               d_world_g[1] + r_k * cols, // preskoči zgornji halo
               half1 * cols * sizeof(float),
               cudaMemcpyDeviceToHost);

    // =========================================================================
    // ČIŠČENJE
    // =========================================================================
    for (int g = 0; g < num_gpus; g++)
    {
        cudaSetDevice(g);
        cudaFree(d_w[g]);
        cudaFree(d_world_g[g]);
        cudaFree(d_next_world_g[g]);
    }

    free(w);
    return world;
}
