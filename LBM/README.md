# LBM for Phase Change

The implementation is based on : *Phase-change modeling based on a novel conservative phase-field method (Reza Haghani-Hassan-Abadi et al. 2021)*

## Simulating Results

Here are two benchmarks from the paper. In videos, gray domain is vapor and blue domain is water. The second and third block represent magnitude of velocity and temperature.

### Benchmark 1
This benchmark is bubble growth with a constant mass flux:

https://github.com/user-attachments/assets/4bf037e9-b9c7-4f1f-9753-364dade03081

The change of radius:

<img src="https://github.com/user-attachments/assets/a9f58b3e-2afd-42c3-be70-92327c4e7098" style="width:50%">

The distribution of velocity and $\phi$ when $R/R(0) = 2$:
![download](https://github.com/user-attachments/assets/049d5f2f-158f-4fd4-9d9d-f2f91c55961c)

Results from the paper:
![download (1)](https://github.com/user-attachments/assets/4b0803e0-4c36-4587-82c8-52f96c8b25d5)


We can see that the increase of radius is linear and the distribution of $\phi$ is correct. But the velocity has some artifacts.

### Benchmark 2
This bench mark is evaporation of liquid drop.

https://github.com/user-attachments/assets/35cfa72f-3042-4173-8842-bc6a51aecd43

With different $h_{fg}$ (latent heat of evaporation), the speed of evaporation is different.

![Figure_4](https://github.com/user-attachments/assets/cb3388cd-3041-4c4c-8e35-6c7856c9bc4a)

They follow the $d^2$ law.

Additionally, if we initialize the liquid drop as a square:

https://github.com/user-attachments/assets/34d0f37c-bf8f-4cd7-a18f-bd56a8e58848


