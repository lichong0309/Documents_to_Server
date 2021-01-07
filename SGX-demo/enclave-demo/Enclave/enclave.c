#include <stdio.h>
#include "enclave_t.h"

char * secret = "I am the SGX secret\n";
char buff[100]; // used to be copied outside of Enclave

void e_call_print_secret() {
    char * secret = "modify\n";
    snprintf(buff,100,"%s",secret);
    // printf("%s\n",secret);
    ocall_print_string(buff);
}
