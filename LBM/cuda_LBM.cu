#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <curand_kernel.h>
#include <ctime>
#include <cmath>

#define NX 301
#define NY 301
#define BLOCK_SIZE 16
#define POS(x, y) (y * NX + x)
#define POSK(x, y, k) ((y * NX + x) * 9 + k)

__constant__ float W = 4.0f;
__constant__ float rho_l = 999.9f;
__constant__ float rho_v = 99.9f;
__constant__ float sigma = 1e-4;
__constant__ float tau_phi = 0.8;
__constant__ float tau = 0.8;


__constant__ float w[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

__constant__ int2 e[9] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1},
                      {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

// CUDA device arrays
float *rho, *phi;
float2 *vel;
float *old_temperature, *new_temperature;

float *h_old, *h_new;
float *g_old, *g_new;
float *p_star;

float2 *phi_grad, *rho_grad;
float *phi_laplacian, *vel_div;

char *show_buffer;
// Host arrays
char show[3 * 3 * NX * NY];


__device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}


__device__ float2 i2f(int2 a) {
    return {float(a.x), float(a.y)};
}


__device__ float GetTempPos(int x, int y, float *temp) {
    if ((x==0) || (x==NX-1) || (y==0) || (y==NY-1))
        return 1000.0f;
    else
        return temp[POS(x, y)];
}


__device__ float hEq(int i, int j, int k, float2 *vel, float *phi){
    float eu = dot(i2f(e[k]), vel[POS(i, j)]);
    float uv = dot(vel[POS(i, j)], vel[POS(i, j)]);
    float res = w[k] * phi[POS(i, j)] * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * uv);
    return res;
}


__device__ float gEq(int i, int j, int k, float2 *vel, float *p){
    float eu = dot(i2f(e[k]), vel[POS(i, j)]);
    float uv = dot(vel[POS(i, j)], vel[POS(i, j)]);
    float res = w[k] * (p[POS(i, j)] + 3.0f * eu + 4.5f * eu * eu - 1.5f * uv);
    return res;
}

// __global__ void InitWeight(float *w, int2 *e) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx >= 9) return;

//     float w_vals[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
//                         1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

//     int2 e_vals[9] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1},
//                       {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

//     w[idx] = w_vals[idx];
//     e[idx] = e_vals[idx];
// }


// CUDA Kernel to Initialize Data
__global__ void InitField(float *rho, float *phi, float *h, float *g, float *p, float2 *vel, float *old_temp, float *new_temp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // initialize rho, phi
    int mid_x = NX / 2, mid_y = NY / 2;
    float r_init = float(NX) / 10.0f;

    if (((i-mid_x) * (i-mid_x) + (j-mid_y) * (j-mid_y)) < r_init * r_init) {
        phi[POS(i, j)] = 0.0f;
        rho[POS(i, j)] = rho_v;
    }
    else{
        phi[POS(i, j)] = 1.0f;
        rho[POS(i, j)] = rho_l;
    }

    // initialize h, g, p
    p[POS(i, j)] = 0.0f;
    vel[POS(i, j)] = {0.0f, 0.0f};
    for (int k = 0; k < 9; k++) {
        h[POSK(i, j, k)] = hEq(i, j, k, vel, phi);
        g[POSK(i, j, k)] = gEq(i, j, k, vel, p);
        p[POS(i, j)] += g[POSK(i, j, k)];
    }

    // initialize temperature
    if (i==0 || i==NX-1 || j==0 || j==NY-1) {
        old_temp[POS(i, j)] = new_temp[POS(i, j)] = 1000.0f;
    }
    else {
        old_temp[POS(i, j)] = new_temp[POS(i, j)] = 373.15f;
    }

}


// CUDA Kernel for Temperature Update
__global__ void UpdateTemperature(float *old_temp, float *new_temp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;
    if (i==0 || i==NX-1 || j==0 || j==NY-1) {
        new_temp[POS(i, j)] = 1000.0f;
        return;
    }

    // shuffle order
    curandState state;
    curand_init(1234, threadIdx.x, 0, &state);
    int order[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int idx = 7; idx > 0; --idx) {
        int nidx = curand(&state) % (idx + 1);
        int temp = order[nidx];
        order[nidx] = order[idx];
        order[idx] = temp;
    }

    float laplacian = 0.0f;
    float X = 0.05f;
    int inv[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    for (int idx_k = 0; idx_k < 8; idx_k++) {
        int k = order[idx_k];
        int nx = (i + e[k].x + NX) % NX;
        int ny = (j + e[k].y + NY) % NY;
        int bx = (i + e[inv[k]].x + NX) % NX;
        int by = (j + e[inv[k]].y + NY) % NY;

        float T1 = old_temp[POS(nx, ny)];
        float T2 = old_temp[POS(bx, by)];

        float new_lap = 3.0f * w[k] * (T1 + T2 - 2.0f * old_temp[POS(i, j)]);
        laplacian += new_lap;

        // if (
        //     (i==3)&&(j==6) ||
        //     (i==3)&&(j==294) ||
        //     (i==297)&&(j==6) ||
        //     (i==297)&&(j==294)
        // ){
        //     printf("break %d %d, %d %f %f\n", i, j, k, new_lap, laplacian);
        // }
    }


    new_temp[POS(i, j)] = old_temp[POS(i, j)] + X * laplacian;
}


__global__ void UpdateImg(float *phi, char *show, float2 *vel, float *old_temp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // show[POS(i, j)] = phi[POS(i, j)] * 255;
    float3 col_l = {0.0f, 0.0f, 0.8f};
    float3 col_v = {0.5f, 0.5f, 0.5f};
    // float3 phi_color = phi[POS(i, j)] * col_l + (1.0f - phi[POS(i, j)]) * col_v;
    show[POS(i, j) * 3 + 2] = char(255.f * (phi[POS(i, j)] * col_l.x + (1.0f - phi[POS(i, j)]) * col_v.x));
    show[POS(i, j) * 3 + 1] = char(255.f * (phi[POS(i, j)] * col_l.y + (1.0f - phi[POS(i, j)]) * col_v.y));
    show[POS(i, j) * 3 + 0] = char(255.f * (phi[POS(i, j)] * col_l.z + (1.0f - phi[POS(i, j)]) * col_v.z));

    float vel_mag = sqrtf(dot(vel[POS(i, j)], vel[POS(i, j)])) * 100.0f;
    vel_mag = min(max(vel_mag, 0.0), 1.0);
    show[POS(i, j) * 3 + 3 * NX * NY + 2] = char(255.f * vel_mag);
    show[POS(i, j) * 3 + 3 * NX * NY + 1] = char(255.f * vel_mag);
    show[POS(i, j) * 3 + 3 * NX * NY + 0] = 0.0;

    float temp_scale = (old_temp[POS(i, j)] - 300.f) / 500.0;
    temp_scale = min(max(temp_scale, 0.0), 1.0);
    show[POS(i, j) * 3 + 6 * NX * NY + 2] = char(255.f * temp_scale);
    show[POS(i, j) * 3 + 6 * NX * NY + 1] = 0.0;
    show[POS(i, j) * 3 + 6 * NX * NY + 0] = 0.0;
}

// Swap buffers
void swapBuffers(float *&a, float *&b) {
    float *temp = a;
    a = b;
    b = temp;
}


void Initialize() {
    // Allocate GPU Memory
    cudaMalloc(&rho, NX * NY * sizeof(float));
    cudaMalloc(&phi, NX * NY * sizeof(float));
    cudaMalloc(&vel, NX * NY * sizeof(float2));
    cudaMalloc(&old_temperature, NX * NY * sizeof(float));
    cudaMalloc(&new_temperature, NX * NY * sizeof(float));

    cudaMalloc(&h_old, 9 * NX * NY * sizeof(float));
    cudaMalloc(&h_new, 9 * NX * NY * sizeof(float));
    cudaMalloc(&g_old, 9 * NX * NY * sizeof(float));
    cudaMalloc(&g_new, 9 * NX * NY * sizeof(float));
    cudaMalloc(&p_star, NX * NY * sizeof(float));

    cudaMalloc(&phi_grad, NX * NY * sizeof(float2));
    cudaMalloc(&rho_grad, NX * NY * sizeof(float2));
    cudaMalloc(&phi_laplacian, NX * NY * sizeof(float));
    cudaMalloc(&vel_div, NX * NY * sizeof(float));

    cudaMalloc(&show_buffer, 3 * 3 * NX * NY * sizeof(char));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((NX + BLOCK_SIZE - 1) / BLOCK_SIZE, (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
    InitField<<<grid, block>>>(rho, phi, h_old, g_old, p_star, vel, old_temperature, new_temperature);
    cudaDeviceSynchronize();
}


bool CheckSym(float *array) {
    bool flag = true;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int si = NX - i - 1, sj = NY - j - 1;

            if ((array[POS(i, j)] - array[POS(i, sj)] > 1e-2)
                || (array[POS(i, j)] - array[POS(si, j)] > 1e-2)
                || (array[POS(i, j)] - array[POS(si, sj)] > 1e-2)) {
                    std::cout << "wtf:" << i << " " << j << std::endl;
                    flag = false;
                }
        }
    }

    return flag;
}


int main() {
    Initialize();
    std::cout << "finish initialization!" << std::endl;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((NX + BLOCK_SIZE - 1) / BLOCK_SIZE, (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int frame = 1; frame > 0; frame++) {
        UpdateTemperature<<<grid, block>>>(old_temperature, new_temperature);
        cudaDeviceSynchronize();

        // CalDerivative<<<grid, block>>>()

        // Swap Buffers
        swapBuffers(old_temperature, new_temperature);


        // render
        UpdateImg<<<grid, block>>>(phi, show_buffer, vel, old_temperature);
        cudaDeviceSynchronize();
        cudaMemcpy(show, show_buffer, 3 * 3 * NX * NY * sizeof(char), cudaMemcpyDeviceToHost);
        cv::Mat img(3 * NX, NY, CV_8UC3, show);
        cv::imshow("Temperature Field", img);
        cv::waitKey(1);
    }

    cudaFree(rho);
    cudaFree(phi);
    cudaFree(vel);
    cudaFree(old_temperature);
    cudaFree(new_temperature);
    cudaFree(h_old);
    cudaFree(h_new);
    cudaFree(g_old);
    cudaFree(g_new);
    cudaFree(p_star);
    cudaFree(phi_grad);
    cudaFree(rho_grad);
    cudaFree(phi_laplacian);
    cudaFree(vel_div);
    cudaFree(show_buffer);

    return 0;
}
