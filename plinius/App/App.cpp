/*
在飞地中调用外面的函数叫做ocalls
void ocall_start_clock()                                                    // 获得ocall时钟开始  
void ocall_stop_clock()                                                 // 获得ocall时钟结束   
void ocall_add_loss()                                                       // 
void data_malloc(size_t chunk)
void ocall_read_disk_chunk()
void train_mnist(char *cfgfile)
void read_all_mnist_data()
void test_mnist(char *cfgfile)
void my_ocall_close()
int initialize_enclave(void)
int SGX_CDECL main(int argc, char *argv[])
*/

#include <stdio.h>
#include <string.h>
#include <pwd.h>
#include <thread>

#include <sgx_urts.h>
#include "App.h"
#include "Enclave_u.h"
#include "ErrorSupport.h"
#include "Romulus_helper.h"

/* For romulus */
#define MAX_PATH FILENAME_MAX

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

/* Romulus objects */
extern PersistentHeader *per_out;
extern uint8_t *base_addr;
extern uint8_t *real_base_addr;

/* Darknet variables */
data training_data, test_data;                          // darknet数据的结构体struct data
size_t chunk_size;                                          

/* Benchmarking */
#include "benchtools.h"
#include <time.h>
struct timespec start, stop;                        // 使用clock_gettime()函数，存储当前的时间
double diff;

comm_info *comm_out;

//---------------------------------------------------------------------------------
/**
 * Config files
 */
#define IMG_SIZE 784

/* #define CONFIG_FILE "./App/dnet-out/cfg/mnist.cfg"
#define MNIST_TRAIN_IMAGES "./App/dnet-out/data/mnist/train-images-idx3-ubyte" 
#define MNIST_TRAIN_LABELS "./App/dnet-out/data/mnist/train-labels-idx1-ubyte" */

// 测试集数据的文件位置：images(data) 和labels
#define MNIST_TEST_IMAGES "./App/dnet-out/data/mnist/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS "./App/dnet-out/data/mnist/t10k-labels-idx1-ubyte"

//训练集数据的文件位置： images(data)和labels
#define MNIST_TRAIN_IMAGES "./App/dnet-out/data/mnist/enc_mnist_imgs.data"
#define MNIST_TRAIN_LABELS "./App/dnet-out/data/mnist/enc_mnist_labels.data"


// darknet框架神经网络的配置文件mnist.cfg路径
#define MNIST_CFG "./App/dnet-out/cfg/cifar.cfg"


/* For benchmarking */
/**
 * We can't measure time in the enclave runtime so we measure it with an ocall.
 * The transition time is very small(~5ns) compared to the times we measure (ms,s,mins etc),
 * so the values are accurate enough for our use cases
 */
void ocall_start_clock()                    // 存储当前时间到 struct timespect * start中，作为程序的开始的时间点
{
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);           
    // int clock_getting(clockid_t clk_id, struct timespect *tp);
    // clockid_t clk_id用于指定计时时钟的类型。
    // CLOCK_MONOTONIC_RAW: 本地振荡器，不受NTP约束。
}   
void ocall_stop_clock()                         // 存储当前时间到strcut timespect *stop中，作为程序的结束的时间点                     
{
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
}

void ocall_add_loss()
{
    //you can use this routine to register the loss and iteration during training in a file
    //update the loss in the enclave in comm_in->loss and the iteration in comm_in->epoch
}

//alloc space for mnist training data variable
void data_malloc(size_t chunk)
{
    //removed
}

//This ocall reads encrypted mnist data from disk to DRAM
//将加密的minist训练集放到DRAM中
void ocall_read_disk_chunk()
{

    printf("Reading initial training data from disk\n");
    if (&training_data != NULL)
    {
        free_data(training_data);
    }
    std::string img_path = MNIST_TRAIN_IMAGES;                              // 定义路径
    std::string label_path = MNIST_TRAIN_LABELS;                            // 定义路径
    //load encrypted data into volatile matrix  
    training_data = load_enc_mnist_images(img_path, NUM_IMGS_MNIST);                        // 加载加密的images
    training_data.y = load_enc_mnist_labels(label_path, NUM_IMGS_MNIST);                      // 加载加密的labels
    comm_out->data_chunk = &training_data;            
    printf("Done reading disk data\n");
    //ecall_set_data(global_eid, &training_data);
}


// 训练数据
void train_mnist(char *cfgfile)
{
    list *config_sections = read_cfg(cfgfile);                          // 读取cfgfile配置文件
    comm_out = (comm_info *)malloc(sizeof(comm_info));                         
    comm_out->config = config_sections;
    ecall_trainer(global_eid,comm_out->config, &training_data, chunk_size, comm_out);       //调用enclave中的可信函数，训练数据
    printf("Mnist training complete..\n");
    free_data(training_data);                       //darknet框架函数，释放内存
}

void read_all_mnist_data()
{
    // fill it
}

/**
 * Test a trained mnist model
 * 测试数据函数
 */
void test_mnist(char *cfgfile)
{
    std::string img_path = MNIST_TEST_IMAGES;                   // 定义测试images的路径
    std::string label_path = MNIST_TEST_LABELS;                 // 定义测试labels的路径
    data test = load_mnist_images(img_path, 10000);         // 加载测试数据 数据x, load_mnist_images(): /dnet-out/src/data_mnist.cpp模块函数
    test.y = load_mnist_labels(label_path, 10000);              // 加载测试数据  labelsy，load_mnist_images():  /dnet-out/src/data_mnist.cpp模块函数
    list *config_sections = read_cfg(cfgfile);                              // 读取cfgfile配置文件, read_cfg(): /dnet-out/src/parser.c模块函数


    // void ecall_tester(list *sections, data *test_data, int pmem)
    // global_eid : int pmem : 0 
    // config_sections : list * sections
    // test : data
    ecall_tester(global_eid, config_sections, &test, 0);            // 调用enclave中的可信函数，对模型进行测试

    printf("Mnist testing complete..\n");
    free_data(test);                                                        // darknet框架函数，释放内存
}

//--------------------------------------------------------------------------------------------------------------

/* Do munmap and close file */
void my_ocall_close()
{
    close_file();
}

/* Initialize the enclave:
 * Call sgx_create_enclave to initialize an enclave instance
 */
// 初始化enclave
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS)
    {
        print_error_message(ret);
        return -1;                              // error则返回-1
    }

    return 0;                                   // 执行正确则返回0
}

/* Application entry */
// 函数入口
int SGX_CDECL main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    sgx_status_t ret;

    /* Initialize the enclave */
    if (initialize_enclave() < 0)
    {
        printf("Enter a character before exit ...\n");
        getchar();
        return -1;
    }
    //Initialize sgx-rom
    rom_init();                     
    printf("Base addr is : %p\n", base_addr);
    ecall_init(global_eid, (void *)per_out, base_addr);

    //mnist model config file
    char cfg[128] = MNIST_CFG;                                   // 神经网络的配置文件

    // 训练数据部分
    //train a model on mnist via the Plinius workflow
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);                             // 获得当前时钟，存放到start中，记作为训练开始的时钟
    train_mnist(cfg);                                                                                           // 训练数据，神经网络配置文件cfg = MNIST_CFG= "./App/dnet-out/cfg/mnist.cfg"
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);                                      // 获取当前时钟，存放到stop中，记作为训练结束的时钟
    printf("Total training time: %f mins\n", time_diff(&start, &stop, SEC) / 60);                                                   // 打印训练数据所需要的时间


    // 测试数据部分
    //test the accuracy of the trained model
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);                                 // 获得当前时钟，存放在start中，记作为测试开始的时钟                         
    test_mnist(cfg);                                                                                                    // 测试数据,神经网络配置文件cfg = MNIST_CFG= "./App/dnet-out/cfg/mnist.cfg"
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);                                      //  获取当前时钟，存放到stop中，就作为测试结束的时钟
    printf("Total inference time: %f mins\n", time_diff(&start, &stop, SEC) / 60);                          // 打印测试数据所需要的时间

    //Destroy enclave
    // 程序执行完成，销毁enclave
    sgx_destroy_enclave(global_eid);                            
    return 0;
}
