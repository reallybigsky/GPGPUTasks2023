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

template <typename FUNC, typename ID, typename RES, std::enable_if_t<std::is_fundamental<RES>::value, bool> = true>
RES OCL_CALL(std::string&& filename, int line, FUNC&& func, ID id, int cl_macro)  {
    RES result;
    reportError(std::forward<FUNC>(func)(id, cl_macro, sizeof(result), &result, nullptr), filename, line);
    return result;
}

template <typename FUNC, typename ID, typename RES, typename = std::void_t<decltype(std::declval<RES>().size()), decltype(std::declval<RES>().data())>>
RES OCL_CALL(std::string&& filename, int line, FUNC&& func, ID id, int cl_macro) {
    size_t size = 0;
    reportError(std::forward<FUNC>(func)(id, cl_macro, 0, nullptr, &size), filename, line);
    RES result(size, 0);
    reportError(std::forward<FUNC>(func)(id, cl_macro, size, result.data(), nullptr), filename, line);
    return result;
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_SAFE_CALL_INFO(RES, FUNC, ID, ...) OCL_CALL<decltype(FUNC), decltype(ID), RES>(__FILE__, __LINE__, FUNC, ID, __VA_ARGS__)

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        std::vector<unsigned char> platformName = OCL_SAFE_CALL_INFO(std::vector<unsigned char>, clGetPlatformInfo, platform, CL_PLATFORM_NAME);
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
        std::vector<unsigned char> platformVendorName = OCL_SAFE_CALL_INFO(std::vector<unsigned char>, clGetPlatformInfo, platform, CL_PLATFORM_VENDOR);
        std::cout << "\tPlatform vendor: " << platformVendorName.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "\tNumber of devices of this OpenCL platform: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::cout << "\t\tDevice #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

            // DEVICE_NAME
            std::vector<unsigned char> deviceName = OCL_SAFE_CALL_INFO(std::vector<unsigned char>, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_NAME);
            std::cout << "\t\t\tDevice name: " << deviceName.data() << std::endl;

            // DEVICE_TYPE
            cl_device_type deviceType = OCL_SAFE_CALL_INFO(cl_device_type, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_TYPE);
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
            cl_ulong deviceMemSize = OCL_SAFE_CALL_INFO(cl_ulong, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE);
            std::cout << "\t\t\tDevice memory size: " << deviceMemSize / (1 << 20) << " MB" << std::endl;

            // DEVICE_DRIVER_VERSION
            std::vector<unsigned char> deviceDriver = OCL_SAFE_CALL_INFO(std::vector<unsigned char>, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_VERSION);
            std::cout << "\t\t\tDevice driver version: " << deviceDriver.data() << std::endl;

            // DEVICE_MAX_CLOCK_FREQUENCY
            size_t deviceMaxWorkGroupSize = OCL_SAFE_CALL_INFO(size_t, clGetDeviceInfo, devices[deviceIndex], CL_DEVICE_MAX_WORK_GROUP_SIZE);
            std::cout << "\t\t\tDevice max work group size: " << deviceMaxWorkGroupSize << std::endl;
        }
    }

    return 0;
}
