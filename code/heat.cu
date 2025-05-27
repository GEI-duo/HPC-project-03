#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BMP_HEADER_SIZE 54
#define ALPHA 0.01 
#define L 0.2      
#define DX 0.02    
#define DY 0.02    
#define DT 0.0005  
#define T 1500     

#define CHECK_CUDA_ERRORS                                       \
    cudaError_t err = cudaGetLastError();                       \
    if (err != cudaSuccess) {                                   \
        printf("CUDA Error: %s\n", cudaGetErrorString(err));    \
    }                                                           \

__global__ 
void initialize_grid_kernel(
    double *grid, 
    int nx, 
    int ny
)
{
    // Initialize the grid diagonals to be heat sources.
    // 
    // Steps:
    //  1. Identify executing code with i/j.
    //  2. Initialize the diagonal heat grid, excluding the matrix borders.
    
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1,
        j = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure safe CUDA memory access
    if (i > nx || j >= ny) return;

    int is_edge     = i == 0 || i == nx - 1 || j == 0 || j == ny - 1, 
        is_diagonal = i == j || i == nx - j - 1;

    grid[i * ny + j] = !is_edge && is_diagonal ? T : 0.0;
}

void initialize_grid(
    double *grid,
    int nx,
    int ny
)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((ny + blockDim.x - 1) / blockDim.x, (nx + blockDim.y - 1) / blockDim.y);
    
    CHECK_CUDA_ERRORS;

    initialize_grid_kernel<<<gridDim, blockDim>>>(grid, nx, ny);
    cudaDeviceSynchronize();
}

__global__ 
void heat_kernel(
    double* grid, 
    double* new_grid, 
    int nx, 
    int ny, 
    double r
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1) return;

    int idx = i * ny + j;

    new_grid[idx] = grid[idx]
        + r * (grid[idx + ny] + grid[idx - ny] - 2.0 * grid[idx])  // vertical
        + r * (grid[idx + 1] + grid[idx - 1] - 2.0 * grid[idx]);   // horizontal
}

__global__ 
void apply_boundary_conditions(
    double* grid, 
    int nx, 
    int ny
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny) {
        grid[0 * ny + idx] = 0.0;
        grid[(nx - 1) * ny + idx] = 0.0;
    }

    if (idx < nx) {
        grid[idx * ny + 0] = 0.0;
        grid[idx * ny + (ny - 1)] = 0.0;
    }
}

void solve_heat_equation(
    double *d_grid, 
    double *d_new_grid, 
    int nx,
    int ny,
    int steps,
    double r
)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((ny + blockDim.x - 1) / blockDim.x, (nx + blockDim.y - 1) / blockDim.y);
    CHECK_CUDA_ERRORS;

    for (int step = 0; step < steps; step++)
    {
        heat_kernel<<<gridDim, blockDim>>>(d_grid, d_new_grid, nx, ny, r);
        cudaDeviceSynchronize();
        apply_boundary_conditions<<<gridDim, blockDim>>>(d_grid, nx, ny);
        cudaDeviceSynchronize();

        double *temp = d_grid;
        d_grid = d_new_grid;
        d_new_grid = temp;
    } 
}

void write_bmp_header(
    FILE *file, 
    int width, 
    int height
)
{
    unsigned char header[BMP_HEADER_SIZE] = {0};

    int file_size = BMP_HEADER_SIZE + 3 * width * height;
    header[0] = 'B';
    header[1] = 'M';
    header[2] = file_size & 0xFF;
    header[3] = (file_size >> 8) & 0xFF;
    header[4] = (file_size >> 16) & 0xFF;
    header[5] = (file_size >> 24) & 0xFF;
    header[10] = BMP_HEADER_SIZE;

    header[14] = 40; // Info header size
    header[18] = width & 0xFF;
    header[19] = (width >> 8) & 0xFF;
    header[20] = (width >> 16) & 0xFF;
    header[21] = (width >> 24) & 0xFF;
    header[22] = height & 0xFF;
    header[23] = (height >> 8) & 0xFF;
    header[24] = (height >> 16) & 0xFF;
    header[25] = (height >> 24) & 0xFF;
    header[26] = 1;  // Planes
    header[28] = 24; // Bits per pixel

    fwrite(header, 1, BMP_HEADER_SIZE, file);
}

__device__ // GPU only function !! 
void get_color(
    double value, 
    unsigned char *r, 
    unsigned char *g, 
    unsigned char *b
)
{

    if (value >= 500.0)
    {
        *r = 255;
        *g = 0;
        *b = 0; // Red
    }
    else if (value >= 100.0)
    {
        *r = 255;
        *g = 128;
        *b = 0; // Orange
    }
    else if (value >= 50.0)
    {
        *r = 171;
        *g = 71;
        *b = 188; // Lilac
    }
    else if (value >= 25)
    {
        *r = 255;
        *g = 255;
        *b = 0; // Yellow
    }
    else if (value >= 1)
    {
        *r = 0;
        *g = 0;
        *b = 255; // Blue
    }
    else if (value >= 0.1)
    {
        *r = 5;
        *g = 248;
        *b = 252; // Cyan
    }
    else
    {
        *r = 255;
        *g = 255;
        *b = 255; // white
    }
}

__global__ 
void prepare_pixel_data(
    double* grid,
     unsigned char* pixel_data,
     int nx,
     int ny,
     int padded_row_size,
     int padding
) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y,
        j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx || j >= ny) return;

    int row_index       = nx - 1 - i,
        row_offset      = i * padded_row_size,
        pixel_offset    = row_offset + j * 3;

    unsigned char r, g, b;
    get_color(grid[row_index * ny + j], &r, &g, &b);

    pixel_data[pixel_offset + 0] = b;
    pixel_data[pixel_offset + 1] = g;
    pixel_data[pixel_offset + 2] = r;

    // Only one thread per row handles padding
    if (j == 0 && padding > 0) {
        for (int p = 0; p < padding; ++p) {
            pixel_data[row_offset + ny * 3 + p] = 0;
        }
    }
}


void write_grid(
    FILE* file, 
    double* d_grid, 
    int nx, 
    int ny
) {
    int row_stride = ny * 3;
    int padding = (4 - (row_stride % 4)) % 4;
    int padded_row_size = row_stride + padding;
    int total_size = nx * padded_row_size;

    // Allocate output buffer on GPU
    unsigned char* d_pixel_data;
    cudaMalloc(&d_pixel_data, total_size);

    dim3 blockDim(16, 16);
    dim3 gridDim((ny + blockDim.x - 1) / blockDim.x, (nx + blockDim.y - 1) / blockDim.y);
    CHECK_CUDA_ERRORS;

    prepare_pixel_data<<<gridDim, blockDim>>>(d_grid, d_pixel_data, nx, ny, padded_row_size, padding);
    cudaDeviceSynchronize();

    // Copy back to host
    unsigned char* h_pixel_data = (unsigned char*)malloc(total_size);
    cudaMemcpy(h_pixel_data, d_pixel_data, total_size, cudaMemcpyDeviceToHost);

    fwrite(h_pixel_data, 1, total_size, file);

    cudaFree(d_pixel_data);
    free(h_pixel_data);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: heat_mpi <size> <steps> <name_output_file>\n");
        return 1;
    }
    
    double r = ALPHA * DT / (DX * DY);
    int nx, ny, steps;
    float ms = 0;
    
    nx      =
    ny      = atoi(argv[1]),
    steps   = atoi(argv[2]);
    
    double *d_grid, *d_new_grid;
    cudaMalloc(&d_grid, nx * ny * sizeof(double));
    cudaMalloc(&d_new_grid, nx * ny * sizeof(double));
    
    // TIMING :: START
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    initialize_grid(d_grid, nx, ny);
    solve_heat_equation(d_grid, d_new_grid, nx, ny, steps, r);

    // TIMING :: END
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    FILE *file = fopen(argv[3], "wb");
    if (!file){ fprintf(stderr, "Failed to open output file\n");}
    else {
        write_bmp_header(file, nx, ny);
        write_grid(file, d_grid, nx, ny);
        fclose(file);
    }

    printf("Execution Time = %.3fms for %dx%d grid and %d steps\n", ms, nx, ny, steps);

    // FREE
    cudaFree(d_grid);
    cudaFree(d_new_grid);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}