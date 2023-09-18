#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <type_traits>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string& filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

template <typename RES, typename... ARGS>
    requires(std::is_fundamental_v<RES>)
RES OCL_CALL(std::string&& filename, int line, auto&& func, ARGS&&... args)  {
    RES result;
    reportError(std::forward<decltype(func)>(func)(std::forward<ARGS>(args)..., sizeof(result), &result, nullptr), filename, line);
    return result;
}

template <typename RES, typename SIZE, typename... ARGS>
    requires(requires(RES&& tmp) { tmp.size(); tmp.data(); })
RES OCL_CALL(std::string&& filename, int line, auto&& func, ARGS&&... args) {
    SIZE size = 0;
    reportError(std::forward<decltype(func)>(func)(std::forward<ARGS>(args)..., 0, nullptr, &size), filename, line);
    RES result(size, 0);
    reportError(std::forward<decltype(func)>(func)(std::forward<ARGS>(args)..., size, result.data(), nullptr), filename, line);
    return result;
}

#define OCL_SAFE_CALL_OLD(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_SAFE_CALL(RES, ...) OCL_CALL<RES>(__FILE__, __LINE__, __VA_ARGS__)
#define OCL_SAFE_CALL_ARRAY(RES, SIZE, ...) OCL_CALL<RES, SIZE>(__FILE__, __LINE__, __VA_ARGS__)

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    auto platforms = OCL_SAFE_CALL_ARRAY(std::vector<cl_platform_id>, cl_uint, clGetPlatformIDs);
    std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

    for (int platformIndex = 0; platformIndex < platforms.size(); ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platforms.size() << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        auto platformName = OCL_SAFE_CALL_ARRAY(std::string, size_t, clGetPlatformInfo, platform, CL_PLATFORM_NAME);
        std::cout << "\tPlatform name: " << platformName.data() << std::endl;

        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        auto platformVendorName = OCL_SAFE_CALL_ARRAY(std::string, size_t, clGetPlatformInfo, platform, CL_PLATFORM_VENDOR);
        std::cout << "\tPlatform vendor: " << platformVendorName.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        auto devices = OCL_SAFE_CALL_ARRAY(std::vector<cl_device_id>, cl_uint, clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL);
        std::cout << "\tNumber of devices of this OpenCL platform: " << devices.size() << std::endl;

        for (int deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::cout << "\t\tDevice #" << (deviceIndex + 1) << "/" << devices.size() << std::endl;

            // DEVICE_NAME
            auto deviceName = OCL_SAFE_CALL_ARRAY(std::string, size_t, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_NAME);
            std::cout << "\t\t\tDevice name: " << deviceName.data() << std::endl;

            // DEVICE_TYPE
            auto deviceType = OCL_SAFE_CALL(cl_device_type, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_TYPE);
            std::cout << "\t\t\tDevice type: ";
            if (deviceType & CL_DEVICE_TYPE_CPU)         std::cout << "CPU ";
            if (deviceType & CL_DEVICE_TYPE_GPU)         std::cout << "GPU ";
            if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "ACCELERATOR ";
            if (deviceType & CL_DEVICE_TYPE_DEFAULT)     std::cout << "DEFAULT TYPE ";
            if (!(deviceType & (CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_DEFAULT))) {
                std::cout << "SOMETHING STRANGE";
            }
            std::cout << std::endl;

            // DEVICE_MEM_SIZE
            auto deviceMemSize = OCL_SAFE_CALL(cl_ulong, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE);
            std::cout << "\t\t\tDevice memory size: " << deviceMemSize / (1 << 20) << " MB" << std::endl;

            // DEVICE_DRIVER_VERSION
            auto deviceDriver = OCL_SAFE_CALL_ARRAY(std::string, size_t, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_VERSION);
            std::cout << "\t\t\tDevice driver version: " << deviceDriver.data() << std::endl;

            // DEVICE_MAX_CLOCK_FREQUENCY
            auto deviceMaxWorkGroupSize = OCL_SAFE_CALL(size_t, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_MAX_WORK_GROUP_SIZE);
            std::cout << "\t\t\tDevice max work group size: " << deviceMaxWorkGroupSize << std::endl;
        }
    }

    return 0;
}
