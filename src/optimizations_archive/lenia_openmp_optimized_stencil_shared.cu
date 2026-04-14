#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

#include <cuda_runtime.h>
#include <cuda.h>

// Vklop GIFa. Za meritve časa obvezno zakomentiraj, ker cudaMemcpy ob
// vsakem koraku čisto ubije hitrost simulacije (počasno PCIe vodilo).
// #define GENERATE_GIF

// 0 = 32x32 kvadrat (najboljši memory coalescing in najmanj redundantnega branja roba)
// 1 = vodoravni pas, 2 = navpični pas (najslabše za predpomnilnik)
#define BLOCK_CONFIG 0

// Funkcije, ki laufajo izključno na grafični kartici (__device__)
__device__ float gauss_dev(float x, float mu, float sigma)
{
    float diff = (x - mu) / sigma;
    return expf(-0.5f * (diff * diff)); // expf je hitrejši od navadnega exp
}

__device__ float growth_lenia_dev(float u)
{
    float mu = 0.15f;
    float sigma = 0.015f;
    return -1.0f + 2.0f * gauss_dev(u, mu, sigma);
}

// Glavni kernel (__global__), ki ga kliče CPU, izvede pa GPU
__global__ void evolve_lenia_kernel(const float *world, float *next_world, const float *w,
                                    int rows, int cols, int kernel_size, float dt)
{
    // Deljeni pomnilnik (shared memory). Cca 100x hitrejši od VRAM-a.
    // Dinamična velikost, ki jo podamo ob klicu kernela v main zanki.
    extern __shared__ float tile[];

    int r_k = kernel_size / 2;

    // Velikost našega kosa (dimenzija bloka niti + zunanji rob)
    int tileWidth = blockDim.x + 2 * r_k;
    int tileHeight = blockDim.y + 2 * r_k;

    // Kje smo v celotni sliki
    int gj = blockIdx.x * blockDim.x + threadIdx.x;
    int gi = blockIdx.y * blockDim.y + threadIdx.y;

    // Kje smo znotraj našega tile-a (premaknjeno za debelino roba)
    int tx = threadIdx.x + r_k;
    int ty = threadIdx.y + r_k;

    // Skupinsko nalaganje iz počasnega VRAM-a v hitri deljeni pomnilnik.
    // Ker je tile večji od števila niti, mora vsaka nit naložiti več elementov.
    for (int dy = threadIdx.y; dy < tileHeight; dy += blockDim.y)
    {
        for (int dx = threadIdx.x; dx < tileWidth; dx += blockDim.x)
        {
            int src_i = (int)(blockIdx.y * blockDim.y) + dy - r_k;
            int src_j = (int)(blockIdx.x * blockDim.x) + dx - r_k;

            // Toroidalni robovi. Namesto modula (%) uporabimo if-e, ker je hitreje.
            if (src_i < 0)
                src_i += rows;
            else if (src_i >= rows)
                src_i -= rows;
            if (src_j < 0)
                src_j += cols;
            else if (src_j >= cols)
                src_j -= cols;

            tile[dy * tileWidth + dx] = world[src_i * cols + src_j];
        }
    }

    // OBVEZNO! Počakamo, da vse niti naložijo svoje koščke, preden gremo računat.
    __syncthreads();

    // Izračun konvolucije
    if (gi < rows && gj < cols)
    {
        float sum = 0.0f;

        // Zrcaljeno jedro, beremo pa izključno iz hitrega tile-a
        for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
        {
            for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
            {
                int tile_i = ty - r_k + kri;
                int tile_j = tx - r_k + kcj;

                sum += w[ki * kernel_size + kj] * tile[tile_i * tileWidth + tile_j];
            }
        }

        // Loop fusion: rezultat zapišemo takoj po izračunu
        float res;
        if (sum < 1e-6f) // preskočimo prazna območja
        {
            res = world[gi * cols + gj] - dt;
        }
        else
        {
            res = world[gi * cols + gj] + dt * growth_lenia_dev(sum);
        }

        // Omejimo vrednost na [0, 1]
        if (res > 1.0f)
            res = 1.0f;
        else if (res < 0.0f)
            res = 0.0f;

        next_world[gi * cols + gj] = res;
    }
}

// Priprava Gaussovega jedra na procesorju (CPU)
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

// Glavna Host funkcija
float *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps,
                    const float dt, const unsigned int kernel_size,
                    const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif("lenia.gif", cols, rows, inferno_pallete, 8, -1, 0);
#endif

    // Priprava začetnega stanja v navadnem RAMu
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

    // Rezervacija prostora na grafični kartici (VRAM)
    float *d_w, *d_world, *d_next_world;
    size_t world_bytes = rows * cols * sizeof(float);
    size_t w_bytes = kernel_size * kernel_size * sizeof(float);

    cudaMalloc((void **)&d_w, w_bytes);
    cudaMalloc((void **)&d_world, world_bytes);
    cudaMalloc((void **)&d_next_world, world_bytes);

    // Prenos iz RAM v VRAM. To naredimo samo enkrat preden gremo v zanko!
    cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_world, world, world_bytes, cudaMemcpyHostToDevice);

    // Nastavitev mreže za GPU
    dim3 threadsPerBlock;
    if (BLOCK_CONFIG == 0)
        threadsPerBlock = dim3(32, 32); // Kvadrat, najboljše
    else if (BLOCK_CONFIG == 1)
        threadsPerBlock = dim3(256, 4); // Vodoravno
    else
        threadsPerBlock = dim3(1024, 1); // Navpično

    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Izračun dinamične velikosti deljenega pomnilnika za klic kernela
    // Formula: velikost bloka + (2 * rob) na obeh oseh
    int r_k = kernel_size / 2;
    int localMemSize = (threadsPerBlock.x + 2 * r_k) *
                       (threadsPerBlock.y + 2 * r_k) * sizeof(float);

    // Glavna zanka simulacije (računamo direktno na GPU)
    for (unsigned int step = 0; step < steps; step++)
    {
        // Zaženemo kernel
        evolve_lenia_kernel<<<numBlocks, threadsPerBlock, localMemSize>>>(
            d_world, d_next_world, d_w, rows, cols, kernel_size, dt);

        // Zamenjamo kazalce namesto da bi kopirali celotno matriko
        float *d_temp = d_world;
        d_world = d_next_world;
        d_next_world = d_temp;

#ifdef GENERATE_GIF
        // Prenos nazaj za GIF. To mora iti čez PCIe, zato je grozno počasno.
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

    // Šele po vseh korakih prenesemo končni rezultat nazaj na CPU
    cudaMemcpy(world, d_world, world_bytes, cudaMemcpyDeviceToHost);

    // Pucanje VRAMa
    cudaFree(d_w);
    cudaFree(d_world);
    cudaFree(d_next_world);
    free(w);

    return world;
}