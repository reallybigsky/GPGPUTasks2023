#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    const int benchmarkingIters = 10;
    const unsigned int N = 32 * 1024 * 1024;
    std::vector<float> as(N, 0);
    FastRandom r(N);
    for (unsigned int i = 0; i < N; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << N << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (N / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    std::vector<float> copy(as);
    gpu::gpu_mem_32f as_gpu, bs_gpu;
    as_gpu.resizeN(N);
    bs_gpu.resizeN(N);
    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), N);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (N + workGroupSize - 1) / workGroupSize * workGroupSize;
            for (unsigned int M = 1; M < N; M *= 2) {
                merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, N, M);
                as_gpu.swap(bs_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (N / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), N);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < N; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
