// 神经网络训练模块
#include "dnet_sgx_utils.h"
#include "darknet.h"
#include "trainer.h"
#include "mirroring/dnet_mirror.h"
#include "mirroring/nvdata.h"
#include "checks.h"

#define NUM_ITERATIONS 10

comm_info *comm_in = nullptr;
NVData *pm_data = nullptr;
data train;
size_t batch_size = 0;
int count;
//define enc_key; this will be provisioned via remote attestation
unsigned char enc_key[16] = {0x76, 0x39, 0x79, 0x24, 0x42, 0x26, 0x45, 0x28, 0x48, 0x2b, 0x4d, 0x3b, 0x62, 0x51, 0x5e, 0x8f};

//global network model
network *net = nullptr;
NVModel *nv_net = nullptr;

/**
 * Pxxxx
 * The network training avg accuracy should decrease
 * as the network learns
 * Batch size: the number of data samples read for one training epoch/iteration
 * If accuracy not high enough increase max batch
 */

//allocate memory for training data variable
//分配内存给训练数据变量
data data_alloc(size_t batch_size)
{
    data temp;
    temp = {0};
    temp.shallow = 0;
    matrix X = make_matrix(batch_size, IMG_SIZE);
    matrix Y = make_matrix(batch_size, NUM_CLASSES);
    temp.X = X;
    temp.y = Y;
    return temp;
}
void ecall_set_data(data *data)
{
    train = *data;
}
//removes pmem net
void rm_nv_net()
{
    printf("Removing PM model\n");
    nv_net = romuluslog::RomulusLog::get_object<NVModel>(0);
    if (nv_net != nullptr)
    {
        TM_PFREE(nv_net);
        romuluslog::RomulusLog::put_object<NVModel>(0, nullptr);
    }
}
//sets pmem training data: for testing purposes with unencrypted data
void set_nv_data(data *tdata)
{
    pm_data = romuluslog::RomulusLog::get_object<NVData>(1);
    if (pm_data == nullptr)
    {
        pm_data = (NVData *)TM_PMALLOC(sizeof(struct NVData));
        romuluslog::RomulusLog::put_object<NVData>(1, pm_data);
        pm_data->alloc();
    }

    if (pm_data->data_present == 0)
    {
        pm_data->fill_pm_data(tdata);
        printf("---Copied training data to pmem---\n");
    }
    //comm training data to nv data
    //train = (data)malloc(sizeof(data));
    // pm_data->shallow_copy_data(&train);
}
void load_pm_data()
{
    pm_data = romuluslog::RomulusLog::get_object<NVData>(1);
    if (pm_data == nullptr)
    {
        printf("---Allocating PM data---\n");
        pm_data = (NVData *)TM_PMALLOC(sizeof(struct NVData));
        romuluslog::RomulusLog::put_object<NVData>(1, pm_data);
        pm_data->alloc();
    }
    if (pm_data->data_present == 0)
    {
        //ocall to copy encrypted data into enclave
        ocall_read_disk_chunk();
        printf("Copying encrypted training data in PM\n");
        pm_data->fill_pm_data(comm_in->data_chunk);
        printf("---Copied training data to PM---\n");
    }

    return;
}


void get_pm_batch()
{
    pm_data = romuluslog::RomulusLog::get_object<NVData>(1);
    if (pm_data == nullptr)
    {
        printf("No PM data\n");
        abort(); //abort training
    }

    if (count % 5 == 0)
    {
        //print this every 10 iters
        printf("Reading and decrypting batch of: %d from PM\n", batch_size);
    }
    pm_data->deep_copy_data(&train, batch_size);
    //printf("Obtained data batch from PM\n");
}
void ecall_trainer(list *sections, data *training_data, int bsize, comm_info *info)
{
    CHECK_REF_POINTER(sections, sizeof(list));
    CHECK_REF_POINTER(training_data, sizeof(data));
    CHECK_REF_POINTER(info, sizeof(comm_info));
    /**
     * load fence after pointer checks ensures the checks are done 
     * before any assignment 
     */
    sgx_lfence();
    //fill pmem data if absent
    /* if (sections == NULL)
    {
        set_nv_data(training_data);
        return;
    } */

    comm_in = info;
    //rm_nv_net();

    train_mnist(sections, training_data, bsize);
}


/**
 * Training algorithms for different models
 */
// 在enclave中训练算法
void train_mnist(list *sections, data *training_data, int pmem)
{
    //TODO: commer checks
    printf("Training mnist in enclave..\n");

    srand(12345);     
    float avg_loss = 0;                        // 初始化训练精度acc
    float loss = 0;                                    // 初始化训练loss 
    int classes = 10;                               // 数据集的label个数
    int N = 60000; //number of training images                      // 训练神经网络图片的数量                  
    int cur_batch = 0;                                  // 
    float progress = 0;
    count = 0;
    int chunk_counter = 0;

    unsigned int num_params;
    //allocate enclave model
    printf("test_0\n");                                                      
    net = create_net_in(sections);                  //   ../src/parser.c中的函数，在enclave中产生一个神经网络
    printf("test_1\n");
    //mirror in if PM net exists
    nv_net = romuluslog::RomulusLog::get_object<NVModel>(0);                            // 实例化nvmodel            //test_4
    if (nv_net != nullptr)
    {
        //mirror in and resume training
        nv_net->mirror_in(net, &avg_loss);                              //将网络模型参数从持久的内存中放到enclave中         // test_5, test_6
    }

    int epoch = (*net->seen) / N;      
    count = 0;

    num_params = get_param_size(net);                       // 获得神经网络参数数量                     // test_2, test_3
    // 由参数计算神经网络模型的大小：模型的参数按照float形式存储，占4个字节，所以神经网络模型的大小与模型的参数数量有关系。
    comm_in->model_size = (double)(num_params * 4) / (1024 * 1024);                     // 使用参数计算神经网络模型的大小,转换成兆(M)

    printf("Max batches: %d\n", net->max_batches);
    printf("Net batch size: %d\n", net->batch);
    printf("Number of params: %d Model size: %f\n", num_params, comm_in->model_size);

    //set batch size
    batch_size = net->batch;
    //allocate training data
    train = data_alloc(batch_size);
    //load data from disk to PM
    load_pm_data();
    //you can reduce the number of iters to a smaller num just for testing purposes
    //net->max_batches = 10;

    //allocate nvmodel here
    if (nv_net == nullptr) //mirror model absent
    {
        nv_net = (NVModel *)TM_PMALLOC(sizeof(struct NVModel));
        romuluslog::RomulusLog::put_object<NVModel>(0, nv_net);
        nv_net->allocator(net);
        avg_loss = -1; //we are training from 0
    }

    //training iterations
    while ((cur_batch < net->max_batches || net->max_batches == 0))
    {
        count++;
        cur_batch = get_current_batch(net);

        /* Get and decrypt batch of pm data */
        get_pm_batch();

        //one training iteration
        loss = train_network_sgd(net, train, 1);

        if (avg_loss == -1)
        {
            avg_loss = loss;
        }

        avg_loss = avg_loss * .95 + loss * .05;
        epoch = (*net->seen) / N;

        progress = ((double)cur_batch / net->max_batches) * 100;
        if (cur_batch % 5 == 0)
        { //print benchmark progress every 10 iters
            printf("Batch num: %ld, Seen: %.3f: Loss: %f, Avg loss: %f avg, L. rate: %f, Progress: %.2f%% \n",
                   cur_batch, (float)(*net->seen) / N, loss, avg_loss, get_current_rate(net), progress);
        }

        //mirror model out to PM
        nv_net->mirror_out(net, &avg_loss);                     // 将训练好的模型放到PM中
    }

    printf("Done training mnist network..\n");
    free_network(net);                              // 将enclave内的net释放掉
}

// 与外界信息传递的测试函数
float ecall_tester(list *sections, data *test_data, int pmem)
{
    CHECK_REF_POINTER(sections, sizeof(list));
    CHECK_REF_POINTER(test_data, sizeof(data));
    /**
     * load fence after pointer checks ensures the checks are done 
     * before any assignment 
     */
    sgx_lfence();
    float middle_layer_ouput  = test_mnist(sections, test_data, pmem);
    return middle_layer_ouput;                      // 返回float的数组
}

void ecall_classify(list *sections, list *labels, image *im)
{
    CHECK_REF_POINTER(sections, sizeof(list));
    CHECK_REF_POINTER(labels, sizeof(list));
    CHECK_REF_POINTER(im, sizeof(image));
    /**
     * load fence after pointer checks ensures the checks are done 
     * before any assignment 
     */
    sgx_lfence();
    //classify_tiny(sections, labels, im, 5);
}

// /**
//  * Test trained mnist model
//  */
// // 测试神经网络
// // list *sections: list config_sections，网络的配置文件
// // data *test_data: &test，测试数据
// void test_mnist(list *sections, data *test_data, int pmem)
// {

//     if (pmem)
//     {
//         //dummy variable
//     }

//     srand(12345);               // ../src/utils.c中的函数，初始化随机数发生器
//     float avg_loss = 0;                         // 初始化loss
//     float avg_acc = 0;                          // 初始化 acc
//     network *net = create_net_in(sections);                 //     ../src/parser.c中的函数，在enclave中产生一个神经网络
    
//     // 实例化nvmodel
//     nv_net = romuluslog::RomulusLog::get_object<NVModel>(0);                        
//     if (nv_net != nullptr)
//     {
//         nv_net->mirror_in(net, &avg_loss);                      // 将网络模型参数从持久的内存中放到enclave中*********
//         printf("Mirrored net in for testing\n");
//     }

//     if (net == NULL)
//     {
//         printf("No neural network in enclave..\n");
//         return;
//     }
//     srand(12345);

//     printf("-----Beginning mnist testing----\n");
   
//     data test = *test_data;                         //  测试数据
    
//     //获得测试的精度
//     // network_accuracies()函数：../src/network.c中的函数，用来测试神经网络的精度acc
//     // 参数： net：神经网络模型；test : 测试的数据集；
//     float *acc = network_accuracies(net, test, 2);                              
//     avg_acc += acc[0];                      // 得到最终训练的平均精度

//     printf("Accuracy: %f%%, %d images\n", avg_acc * 100, test.X.rows);
//     free_network(net);                                          // 释放神经网络      free_network(net)调用 ../src/parser.c模块中的函数

//     /**
//      * Test mnist multi
//      *
//     float avg_acc = 0;
//     data test = *test_data;
//     image im;

//     for (int i = 0; i < test.X.rows; ++i)
//     {
//          im = float_to_image(28, 28, 1, test.X.vals[i]);

//         float pred[10] = {0};

//         float *p = network_predict(net, im.data);
//         axpy_cpu(10, 1, p, 1, pred, 1);
//         flip_image(im);
//         p = network_predict(net, im.data);
//         axpy_cpu(10, 1, p, 1, pred, 1);

//         int index = max_index(pred, 10);
//         int class = max_index(test.y.vals[i], 10);
//         if (index == class)
//             avg_acc += 1;
        
//        printf("%4d: %.2f%%\n", i, 100. * avg_acc / (i + 1)); //un/comment to see/hide accuracy progress
//     }
//     printf("Overall prediction accuracy: %2f%%\n", 100. * avg_acc / test.X.rows);
//     free_network(net);    
//     */
// }





/**
 * Test trained mnist model
 */
// 测试神经网络
// list *sections: list config_sections，网络的配置文件
// data *test_data: &test，测试数据
float test_mnist(list *sections, data *test_data, int pmem)
{
    unsigned int num_par_test ;

    if (pmem)
    {
        //dummy variable
    }

    srand(12345);               // ../src/utils.c中的函数，初始化随机数发生器
    float avg_loss = 0;                         // 初始化loss
    float avg_acc = 0;                          // 初始化 acc
    network *net = create_net_in(sections);                 //     ../src/parser.c中的函数，在enclave中产生一个神经网络
    

    num_par_test = get_param_size(net);                                      // 获取神经网络参数数量

    // 由参数计算神经网络模型的大小：模型的参数按照float形式存储，占4个字节，所以神经网络模型的大小与模型的参数数量有关系。
    comm_in->model_size = (double)(num_par_test * 4) / (1024 * 1024);                     // 使用参数计算神经网络模型的大小,转换成兆(M)
    printf("Number of params: %d Model size: %f\n", num_par_test, comm_in->model_size);                   // 输出模型的大小

    // 提示
    if (net == NULL)
    {
        printf("No neural network in enclave..\n");
        return 0;
    }
    srand(12345);

    printf("-----Beginning mnist testing----\n");
   
    data test = *test_data;                         //  测试数据
    
    float *middle_layer_output = my_network_predict_data(net, test);
    free_network(net);                                          // 释放神经网络      free_network(net)调用 ../src/parser.c模块中的函数
    return *middle_layer_output ;                            // 返回float数组

}