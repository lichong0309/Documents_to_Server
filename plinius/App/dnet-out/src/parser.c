#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>


#include "blas.h"
#include "list.h"
#include "option_list.h"
#include "parser.h"
#include "utils.h"


list *read_cfg(char *filename);


void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}


// 读取神经网络配置cfg文件
/*
**读取神经网络结构配置文件（.cfg文件）中的配置数据，将每个神经网络层参数读取到每个section结构体（每个section是sections的一个节点），
然后全部插入到list结构体sections中并返回。
** 一个section(options)表示配置文件中的一个字段，也就是网络结构中的一层。因此，一个section将读取并存储某一层的参数以及改层的类型（type）
*/
list *read_cfg(char *filename)
{
    printf("read cfg file.\n");
    FILE *file = fopen(filename, "r");                          // 打开文件
    if(file == 0) file_error(filename);                             // 错误提示
    char *line;
    int nu = 0;                                     // 当前读取行号
    list *options = make_list();                                // options包含所有的神经网络层参数，初始化sections
    section *current = 0;                                                           // 当前读取到某一层
    while((line=fgetl(file)) != 0){                         
        ++ nu;
        strip(line);                                                        // 去除读入行中含有的空格符
        switch(line[0]){
            // 以 '[' 开头的行是一个新的section，其内容是层的类型type
            case '[':   
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':                          // 空行
            case '#':                           // 注释
            case ';':                           // 空行
                free(line);                             // 对于上述三种情况直接释放内存即可
                break;
            default:
            // read_option()函数读取网络结构的数据
            // 返回零说明文件中的数据格式有问题，将会提示错误
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}


