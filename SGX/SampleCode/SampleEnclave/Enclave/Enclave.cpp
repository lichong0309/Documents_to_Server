#include "enclave_t.h"                 //包括edl的同名的_t.h文件
#include "sgx_trts.h"                 //包括可信的头文件
#include <string.h>
void dck_test(char *buf,size_t len)   //拷贝函数
{
    const char *secret="Hello Enclave!";
    if(len>=0)
    {
        memcpy(buf,secret,strlen(secret)+1);
    }
}