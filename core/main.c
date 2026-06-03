#include <stdio.h>
#include <CL/cl.h>


int main() {
    cl_int err = CL_SUCCESS;
    cl_uint n = 0;

    err = clGetPlatformIDs(0, NULL, &n);

    if(err == CL_SUCCESS) 
    {
        printf("Found %u platform\n", n);
    }
    else
    {
        printf("clGetPlatformIDs(%i)\n", err);
    }

}