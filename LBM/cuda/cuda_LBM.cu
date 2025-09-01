#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <curand_kernel.h>
#include <ctime>
#include <cmath>

#define BENCHMARK 2

#define NX 301
#define NY 301
#define BLOCK_SIZE 16
#define POS(x, y) (x * NY + y)
#define POSK(x, y, k) (k * NX * NY + x * NY + y)

__constant__ float W = 4.0f;
__constant__ float rho_l = 999.9f;
__constant__ float rho_v = 99.9f;
__constant__ float sigma = 1e-4;
__constant__ float tau_phi = 0.8;
__constant__ float tau = 0.8;
__constant__ float k_water = 0.68;
__constant__ float k_vapor = 0.0304;


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
float temp_cpu[NX * NY];
float phi_cpu[NX * NY];
float h_cpu[NX * NY];
float g_cpu[NX * NY];

__device__ inline float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__ inline float2& operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

__device__ inline float2& operator*=(float2& a, const float& b) {
    a.x *= b;
    a.y *= b;
    return a;
}

__device__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline float2 operator*(float a, const float2& b) {
    return make_float2(a * b.x, a * b.y);
}

__device__ inline float2 normalize(const float2& v) {
    float length = sqrtf(v.x * v.x + v.y * v.y);
    return (length > 1e-6f) ? make_float2(v.x / length, v.y / length) : make_float2(0.0f, 0.0f);
}


__device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}


__device__ float2 i2f(int2 a) {
    return {float(a.x), float(a.y)};
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


__device__ bool isINB(int i, int j, int k, float *phi) {
    bool res = false;
    int ni = (i + e[k].x + NX) % NX, nj = (j + e[k].y + NY) % NY;
    if ((phi[POS(i, j)] - 0.5f) * (phi[POS(ni, nj)] - 0.5f) < 0.0f) {
        res = true;
    }
    return res;
}


__device__ float GetTempPos(int x, int y, float *temp) {
    if ((x==0) || (x==NX-1) || (y==0) || (y==NY-1))
        return 1000.0f;
    else
        return temp[POS(x, y)];
}


__device__ float GetTemp(int x, int y, int k, float *temp, float *phi) {
    float res = 0.0f;
    int nx = (x + e[k].x + NX) % NX, ny = (y + e[k].y + NY) % NY; 
    if(isINB(x, y, k, phi)) {
        int bx = (x - e[k].x + NX) % NX, by = (y - e[k].y + NY) % NY; 
        float T1 = 373.15;
        float T2 = GetTempPos(bx, by, temp);
        float u = (0.5 - phi[POS(x, y)]) / (phi[POS(nx, ny)] - phi[POS(x, y)]);
        res = (2.0 * T1 + (u - 1.0) * T2) / (1.0 + u);
    }
    else {
        res = GetTempPos(nx, ny, temp);
    }

    return res;
}


__device__ float2 CalRhoGradReal(float x, float y, float2 *rho_grad) {
    int x1 = int(x), y1 = int(y);
    int x2 = x1 + 1, y2 = y1 + 1;
    if(x - floor(x) == 0.0){
        x2 = x1;
    }
    if(y - floor(y) == 0.0){
        y2 = y1;
    }

    float tx = x - float(x1), ty = y - float(y1);

    float2 rho11 = rho_grad[POS(x1, y1)];
    float2 rho12 = rho_grad[POS(x1, y2)];
    float2 rho21 = rho_grad[POS(x2, y1)];
    float2 rho22 = rho_grad[POS(x2, y2)];

    float2 grad_x_1 = (1.0 - tx) * rho11 + tx * rho12;
    float2 grad_x_2 = (1.0 - tx) * rho21 + tx * rho22;
    float2 grad = (1.0 - ty) * grad_x_1 + ty * grad_x_2;

    return grad;
}


__device__ float CalMFlux(int x, int y, int k, float *phi, float *temp) {
    float res = 0.0;
    if(isINB(x, y, k, phi)) {

#if BENCHMARK == 1

        res = 0.5;

#elif BENCHMARK == 2

        int bx = (x - e[k].x + NX) % NX;
        int by = (y - e[k].y + NY) % NY;
        int nx = (x + e[k].x + NX) % NX;
        int ny = (y + e[k].y + NY) % NY;

        float T1 = 373.15;
        float T2 = GetTempPos(bx, by, temp);
        float T1_v = GetTempPos(x, y, temp);

        float u = (0.5 - phi[POS(x, y)]) / (phi[POS(nx, ny)] - phi[POS(x, y)]);
        float T0 = (2.0 * T1 + (u - 1.0) * T2) / (1.0 + u);
        float e_dot_T = ((1.0 + 2.0 * u) * T0 - 4.0 * u * T1_v - (1.0 - 2.0 * u) * T2) * 0.5;

        // if(x == 187 && y == 150) {
        //     printf("res=%f, T0=%f, T1_v=%f, T2=%f, u=%f\n", res, T0, T1_v, T2, u);
        // }


        // reverse direction
        bx = (x + 2 * e[k].x + NX) % NX;
        by = (y + 2 * e[k].y + NY) % NY;
        int cur_x = (x + e[k].x + NX) % NX;
        int cur_y = (y + e[k].y + NY) % NY;
        nx = x;
        ny = y;
        T1 = 373.15;
        T2 = GetTempPos(bx, by, temp);
        T1_v = GetTempPos(cur_x, cur_y, temp);

        u = (0.5 - phi[POS(cur_x, cur_y)]) / (phi[POS(nx, ny)] - phi[POS(cur_x, cur_y)]);
        T0 = (2.0 * T1 + (u - 1.0) * T2) / (1.0 + u);
        float e_dot_T_inv = ((1.0 + 2.0 * u) * T0 - 4.0 * u * T1_v - (1.0 - 2.0 * u) * T2) * 0.5;

        // if(x == 187 && y == 150) {
        //     printf("res=%f, T0=%f, T1_v=%f, T2=%f, u=%f\n\n", res, T0, T1_v, T2, u);
        // }

        float h_fg = 1.6;


        if(phi[POS(x, y)] < 0.5) {
            res = (-k_vapor * e_dot_T + k_water * e_dot_T_inv) / (h_fg * sqrtf(dot(i2f(e[k]), i2f(e[k]))));
        }
        else {
            res = (k_water * e_dot_T - k_vapor * e_dot_T_inv) / (h_fg * sqrtf(dot(i2f(e[k]), i2f(e[k]))));
        }

#endif

    }

    return res;
}


__device__ float CalMRate(int i, int j, float *rho, float2 *rho_grad, float *phi, float *temp) {
    int max_idx = 1000;
    float max_product = -10000.0f;
    float distribute_u = 0.0;

    for (int k = 1; k < 9; k++) {
        if(isINB(i, j, k, phi)) {
            int ni = (i + e[k].x + NX) % NX, nj = (j + e[k].y + NY) % NY;
            float u = (0.5f - phi[POS(i, j)]) / (phi[POS(ni, nj)] - phi[POS(i, j)]);
            // TAG: it can cross the bondary
            float inter_x = float(i) + u * e[k].x, inter_y = float(j) + u * e[k].y;
            float2 p_rho_grad = CalRhoGradReal(inter_x, inter_y, rho_grad); 
            float align = dot(normalize(p_rho_grad), normalize(i2f(e[k])));
            if(align > max_product) {
                max_product = align;
                max_idx = k;
                distribute_u = u;
            }
        }
    }

    float res = 0.0f;
    if(max_idx != 1000) {
        // if (i==180 && j==150) {
        //     int ni = (i + e[max_idx].x + NX) % NX, nj = (j + e[max_idx].y + NY) % NY;
        //     printf("%d, phi0=%f, phi1=%f\n", max_idx, phi[POS(i, j)], phi[POS(ni, nj)]);
        // }
        float flux = CalMFlux(i, j, max_idx, phi, temp);
        res = flux * (1.0 - distribute_u);
    }
    else {
        res = 0.0;
    }


    return res;
}


__device__ float2 CalF(int i, int j, float *phi, float *phi_laplacian,
                        float2 *phi_grad, float *p_star, float *g, float *rho, float *vel_div, float2 *vel, float2 *rho_grad) {
    float beta = 12.0 * sigma / W;
    float kappa = 1.5 * W;
    float phi_p = phi[POS(i, j)];
    float miu = 2.0 * beta * phi_p * (1.0 - phi_p) * (1.0 - 2.0 * phi_p) - kappa * phi_laplacian[POS(i, j)];
    float2 F_s = miu * phi_grad[POS(i, j)];

    float2 F_b = {0.0, 0.0};

    float2 F_p = - (1.0 / 3.0) * p_star[POS(i, j)] * rho_grad[POS(i, j)];

    float niu = (tau - 0.5) / 3.0;
    float2 F_eta = {0.0, 0.0};

    for(int k = 0; k < 9; k++) {
        // TAG: why g_new here?
        F_eta.x += e[k].x * e[k].y * (g[POSK(i, j, k)] - gEq(i, j, k, vel, p_star));
        F_eta.y += e[k].x * e[k].y * (g[POSK(i, j, k)] - gEq(i, j, k, vel, p_star));
    }

    F_eta *= -3.0 * niu / tau;
    F_eta.x *= phi_grad[POS(i, j)].x;
    F_eta.y *= phi_grad[POS(i, j)].y;

    float2 F_a = {0.0f, 0.0f};
    F_a = rho[POS(i, j)] * vel_div[POS(i, j)] * vel[POS(i, j)];

    return F_s + F_b + F_p + F_eta + F_a;
    // return float2({0.0, 0.0});
}


__global__ void InitField(float *rho, float *phi, float *h, float *g, float *p, float2 *vel, float *old_temp, float *new_temp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // initialize rho, phi
    int mid_x = NX / 2, mid_y = NY / 2;
    float r_init = float(NX) / 8.0f;

#if BENCHMARK == 1

    if (((i-mid_x) * (i-mid_x) + (j-mid_y) * (j-mid_y)) < r_init * r_init) {
        phi[POS(i, j)] = 0.0f;
        rho[POS(i, j)] = rho_v;
    }
    else{
        phi[POS(i, j)] = 1.0f;
        rho[POS(i, j)] = rho_l;
    }

#elif BENCHMARK == 2

    if (((i-mid_x) * (i-mid_x) + (j-mid_y) * (j-mid_y)) < r_init * r_init) {
        phi[POS(i, j)] = 1.0f;
        rho[POS(i, j)] = rho_l;
    }
    else{
        phi[POS(i, j)] = 0.0f;
        rho[POS(i, j)] = rho_v;
    }

#endif


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


__global__ void UpdateTemperature(float *old_temp, float *new_temp, float2 *vel, float *phi) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;
    if (i==0 || i==NX-1 || j==0 || j==NY-1) {
        new_temp[POS(i, j)] = 2000.0f;
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

    float X_water = 0.30f;
    float X_vapor = 0.06f;
    float X;
    if (phi[POS(i, j)] < 0.5) {
        X = X_vapor;
    }
    else if (phi[POS(i, j)] == 0.5f) {
        X = 0.5 * (X_vapor + X_water);
    }
    else {
        X = X_water;
    }

    float laplacian = 0.0f;
    float2 Tgrad = {0.0f, 0.0f};
    int inv[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    for (int idx_k = 0; idx_k < 8; idx_k++) {
        int k = order[idx_k];
        // int nx = (i + e[k].x + NX) % NX;
        // int ny = (j + e[k].y + NY) % NY;
        // int bx = (i + e[inv[k]].x + NX) % NX;
        // int by = (j + e[inv[k]].y + NY) % NY;

        float T1 = GetTemp(i, j, k, old_temp, phi);
        float T2 = GetTemp(i, j, inv[k], old_temp, phi);

        laplacian += 3.0f * w[k] * (T1 + T2 - 2.0f * old_temp[POS(i, j)]);
        Tgrad += 1.5f * w[k] * (T1 - T2) * i2f(e[k]);
    }


    new_temp[POS(i, j)] = old_temp[POS(i, j)] + X * laplacian - dot(Tgrad, vel[POS(i, j)]);
}


__global__ void CalDerivative(float *phi, float *rho, float2 *vel, float2 *phi_grad, float2 *rho_grad, float *phi_laplacian, float *vel_div) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // shuffle order
    curandState state;
    curand_init(1234, threadIdx.x, 0, &state);
    int order[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (int idx = 8; idx > 0; --idx) {
        int nidx = curand(&state) % (idx + 1);
        int temp = order[nidx];
        order[nidx] = order[idx];
        order[idx] = temp;
    }

    phi_grad[POS(i, j)] = {0.0, 0.0};
    rho_grad[POS(i, j)] = {0.0, 0.0};
    phi_laplacian[POS(i, j)] = 0.0;
    vel_div[POS(i, j)] = 0.0;

    for(int idx_k = 0; idx_k < 9; idx_k++) {
        int k = order[idx_k];
        int ni = (i + e[k].x + NX) % NX;
        int nj = (j + e[k].y + NY) % NY;

        phi_grad[POS(i, j)] += 3.0f * w[k] * phi[POS(ni, nj)] * i2f(e[k]);
        rho_grad[POS(i, j)] += 3.0f * w[k] * rho[POS(ni, nj)] * i2f(e[k]);
        phi_laplacian[POS(i, j)] += 6.0f * w[k] * (phi[POS(ni, nj)] - phi[POS(i, j)]);
        vel_div[POS(i, j)] += 3.0f * w[k] * dot(normalize(i2f(e[k])), vel[POS(ni, nj)]);
    }
    
    vel_div[POS(i, j)] = 0.5f * (vel[POS((i+1)%NX, j)].x + vel[POS(i, (j+1)%NY)].y - vel[POS((i-1+NX)%NX, j)].x - vel[POS(i, (j-1+NY)%NY)].y);
}


__global__ void Collision(
        float *h_old, float *h_new, float *g_old, float *g_new, 
        float *phi, float2 *phi_grad, float *rho, float2 *rho_grad, float *p_star, 
        float *phi_laplacian, float *vel_div, float *old_temperature, float2 *vel) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // shuffle order
    curandState state;
    curand_init(1234, threadIdx.x, 0, &state);
    int order[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (int idx = 8; idx > 0; --idx) {
        int nidx = curand(&state) % (idx + 1);
        int temp = order[nidx];
        order[nidx] = order[idx];
        order[idx] = temp;
    }

    for(int idx_k = 0; idx_k < 9; idx_k++) {
        int k = order[idx_k];

        // ------ update h -----------

        float heq = hEq(i, j, k, vel, phi);

        // formula (22)
        h_old[POSK(i, j, k)] = h_old[POSK(i, j, k)] - (h_old[POSK(i, j, k)] - heq) / tau_phi;

        float R = 0.0;
        float2 normal_grad = normalize(phi_grad[POS(i, j)]);
        if(!(normal_grad.x==0.0 && normal_grad.y==0.0)) {
            // formula (24)
            R = dot(w[k] * i2f(e[k]), 4.0 * phi[POS(i, j)] * (1.0 - phi[POS(i, j)]) / W * normal_grad);
            if(std::isnan(R)) {
                R = 0.0;
            }
        }

        // formula (25)
        float F = w[k] * (1.0f + 3.0f * (dot(i2f(e[k]), vel[POS(i, j)]) * (tau_phi - 0.5f) / tau_phi));
        F *= -CalMRate(i, j, rho, rho_grad, phi, old_temperature) / rho_l;

        h_old[POSK(i, j, k)] += (2.0f * tau_phi - 1.0) / (2.0 * tau_phi) * R + F;


        // -------- update g --------
        float geq = gEq(i, j, k, vel, p_star);
        g_old[POSK(i, j, k)] -= (g_old[POSK(i, j, k)] - geq) / tau;

        // formula (32)
        float P = w[k] * CalMRate(i, j, rho, rho_grad, phi, old_temperature) * (1.0 / rho_v - 1.0 / rho_l);

        // formula (33)
        float G = 3.0f * w[k] * dot(i2f(e[k]), CalF(i, j, phi, phi_laplacian, phi_grad, p_star, g_old, rho, vel_div, vel, rho_grad)) / rho[POS(i, j)];

        // formula (31)
        g_old[POSK(i, j, k)] += (2.0 * tau - 1.0) / (2.0 * tau) * G + P;

    }

}


__global__ void Advection(float *h_old, float *h_new, float *g_old, float *g_new) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    for(int k = 0; k < 9; k++) {
        int i_source = (i - e[k].x + NX) % NX, j_source = (j - e[k].y + NY) % NY;
        h_new[POSK(i, j, k)] = h_old[POSK(i_source, j_source, k)];
        g_new[POSK(i, j, k)] = g_old[POSK(i_source, j_source, k)];
    }
}


__global__ void CalRho(float *phi, float *h_new, float *h_old, float *g_new, float *rho, float *p_star) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // shuffle order
    curandState state;
    curand_init(1234, threadIdx.x, 0, &state);
    int order[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (int idx = 8; idx > 0; --idx) {
        int nidx = curand(&state) % (idx + 1);
        int temp = order[nidx];
        order[nidx] = order[idx];
        order[idx] = temp;
    }

    phi[POS(i, j)] = 0.0;
    p_star[POS(i, j)] = 0.0;
    for(int idx_k = 0; idx_k < 9; idx_k++) {
        int k = order[idx_k];
        phi[POS(i, j)] += h_new[POSK(i, j, k)];
        p_star[POS(i, j)] += g_new[POSK(i, j, k)];
        h_old[POSK(i, j, k)] = h_new[POSK(i, j, k)];
    }

    rho[POS(i, j)] = phi[POS(i, j)] * rho_l + (1.0f - phi[POS(i, j)]) * rho_v;

}

__global__ void CalVel(float *phi, float *phi_laplacian, float2 *phi_grad, float *p_star, float *g_new, float *rho, float *vel_div,
                        float2 *vel, float2 *rho_grad, float *g_old) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // shuffle order
    curandState state;
    curand_init(1234, threadIdx.x, 0, &state);
    int order[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (int idx = 8; idx > 0; --idx) {
        int nidx = curand(&state) % (idx + 1);
        int temp = order[nidx];
        order[nidx] = order[idx];
        order[idx] = temp;
    }

    float2 FF = CalF(i, j, phi, phi_laplacian, phi_grad, p_star, g_new, rho, vel_div, vel, rho_grad);
    vel[POS(i, j)] = (1.0 / (2.0 * rho[POS(i, j)])) * FF;
    for(int idx_k = 0; idx_k < 9; idx_k++) {
        int k = order[idx_k];
        vel[POS(i, j)] += g_new[POSK(i, j, k)] * i2f(e[k]);
        g_old[POSK(i, j, k)] = g_new[POSK(i, j, k)];
    }
}
        
__global__ void UpdateImg(float *phi, char *show, float2 *vel, float *old_temp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<0 || i >= NX || j<0 || j >= NY) return;

    // show[POS(i, j)] = phi[POS(i, j)] * 255;
    float3 col_l = {0.0f, 0.0f, 0.8f};
    float3 col_v = {0.5f, 0.5f, 0.5f};

    float phi_value = min(1.0f, max(0.0f, phi[POS(i, j)]));

    int pos1 = i * 3 * NY + j;
    int pos2 = i * 3 * NY + j + NY;
    int pos3 = i * 3 * NY + j + 2 * NY;

    // float3 phi_color = phi[POS(i, j)] * col_l + (1.0f - phi[POS(i, j)]) * col_v;
    show[pos1 * 3 + 2] = char(255.f * (phi_value * col_l.x + (1.0f - phi_value) * col_v.x));
    show[pos1 * 3 + 1] = char(255.f * (phi_value * col_l.y + (1.0f - phi_value) * col_v.y));
    show[pos1 * 3 + 0] = char(255.f * (phi_value * col_l.z + (1.0f - phi_value) * col_v.z));

    int pos_mid = (NX/2) * 3 * NY + (NY/2);
    show[pos_mid * 3 + 2] = 0xFF;
    show[pos_mid * 3 + 1] = 0xFF;
    show[pos_mid * 3 + 0] = 0xFF;

    float vel_mag = sqrtf(dot(vel[POS(i, j)], vel[POS(i, j)])) * 1000.0f;
    vel_mag = min(max(vel_mag, 0.0), 1.0);
    show[pos2 * 3 + 2] = char(255.f * vel_mag);
    show[pos2 * 3 + 1] = char(255.f * vel_mag);
    show[pos2 * 3 + 0] = 0.0;

    float temp_scale = (old_temp[POS(i, j)] - 300.f) / 500.0;
    temp_scale = min(max(temp_scale, 0.0), 1.0);
    show[pos3 * 3 + 2] = char(255.f * temp_scale);
    show[pos3 * 3 + 1] = 0.0;
    show[pos3 * 3 + 0] = 0.0;
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

            if ((fabs(array[POS(i, j)] - array[POS(i, sj)]) > 1e-5)
                || fabs((array[POS(i, j)] - array[POS(si, j)]) > 1e-5)
                || fabs((array[POS(i, j)] - array[POS(si, sj)]) > 1e-5)) {
                    // std::cout << array[POS(i, j)] << " " << array[POS(i, sj)] << std::endl;
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
        for (int step = 0; step < 100; step++){

            UpdateTemperature<<<grid, block>>>(old_temperature, new_temperature, vel, phi);
            cudaDeviceSynchronize();
            swapBuffers(old_temperature, new_temperature);

            CalDerivative<<<grid, block>>>(phi, rho, vel, phi_grad, rho_grad, phi_laplacian, vel_div);
            cudaDeviceSynchronize();

            Collision<<<grid, block>>>(h_old, h_new, g_old, g_new, phi, phi_grad, rho, rho_grad, p_star, phi_laplacian, vel_div, old_temperature, vel);
            cudaDeviceSynchronize();

            Advection<<<grid, block>>>(h_old, h_new, g_old, g_new);
            cudaDeviceSynchronize();

            CalRho<<<grid, block>>>(phi, h_new, h_old, g_new, rho, p_star);
            cudaDeviceSynchronize();

            CalDerivative<<<grid, block>>>(phi, rho, vel, phi_grad, rho_grad, phi_laplacian, vel_div);
            cudaDeviceSynchronize();

            CalVel<<<grid, block>>>(phi, phi_laplacian, phi_grad, p_star, g_new, rho, vel_div, vel, rho_grad, g_old);
            cudaDeviceSynchronize();
        }

        // cudaMemcpy(temp_cpu, p_star, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
        // bool sym_flat = CheckSym(temp_cpu);
        // if(!sym_flat){
        //     std::cout << frame << std::endl;
        // }


        // render
        UpdateImg<<<grid, block>>>(phi, show_buffer, vel, old_temperature);
        cudaDeviceSynchronize();
        cudaMemcpy(show, show_buffer, 3 * 3 * NX * NY * sizeof(char), cudaMemcpyDeviceToHost);
        cv::Mat img(NX, 3 * NY, CV_8UC3, show);
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
