#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <numeric>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void call_sum(const std::string& kName, const gpu::WorkSize& ws, const std::vector<unsigned int>& src_vec, int iters = 10) {
    gpu::gpu_mem_32u gpu_src;
    gpu_src.resizeN(src_vec.size());
    gpu_src.writeN(src_vec.data(), src_vec.size());

    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kName);

    bool printLog = false;
    kernel.compile(printLog);

    unsigned int reference_sum = std::accumulate(src_vec.begin(), src_vec.end(), 0);

    timer t;
    for (int iter = 0; iter < iters; ++iter) {
        t.start();
        unsigned int sum = 0;

        uint32_t zero = 0;
        gpu::gpu_mem_32u gpu_res;
        gpu_res.resizeN(1);
        gpu_res.writeN(&zero, 1);

        kernel.exec(ws, gpu_src, gpu_res, (unsigned int)src_vec.size());
        gpu_res.readN(&sum, 1);

        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        t.nextLap();
        t.stop();
    }

    std::string out_prefix = "GPU " + kName + ": ";

    std::cout << out_prefix << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << out_prefix << (src_vec.size()/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

}


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    const int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    const unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            t.start();
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
            t.stop();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            t.start();
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
            t.stop();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu::WorkSize ws(workGroupSize, global_work_size);

        call_sum("sum_baseline", ws, as, benchmarkingIters);
        call_sum("sum_arr_non_coalesced", ws, as, benchmarkingIters);
        call_sum("sum_arr_coalesced", ws, as, benchmarkingIters);
        call_sum("sum_local_mem_main_thread", ws, as, benchmarkingIters);
        call_sum("sum_tree", ws, as, benchmarkingIters);
    }
}
