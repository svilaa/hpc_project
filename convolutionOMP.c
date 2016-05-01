// OMP implementation
//
//  convolution.c
//
//
//  Created by Josep Lluis Lerida on 11/03/15.
//
// This program calculates the convolution for PPM images.
// The program accepts an PPM image file, a text definition of the kernel matrix and the PPM file for storing the convolution results.
// The program allows to define image partitions for processing large images (>500MB)
// The 2D image is represented by 1D vector for chanel R, G and B. The convolution is applied to each channel separately.

#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

#define TRUE 1
#define FALSE 0

// Use STATIC DYNAMIC or GUIDED to change the OMP mode
#define STATIC

// Estructura per emmagatzemar el contingut d'una imatge.
struct imagenppm{
    int height;
    int width;
    char *comment;
    int maxcolor;
    int P;
    int *R;
    int *G;
    int *B;
};
typedef struct imagenppm* DataImage;

// Estructura per emmagatzemar el contingut d'un kernel.
struct structkernel{
    int kernelX;
    int kernelY;
    float *vkern;
};
typedef struct structkernel* kernelData;

//Functions Definition
DataImage initimage(char* name, FILE **fp, int partitions, int halo);
DataImage duplicateImageData(DataImage src, int partitions, int halo);

int readImage(DataImage Img, FILE **fp, int dim, int halosize, long int *position);
int duplicateImageChunk(DataImage src, DataImage dst, int dim);
int initfilestore(DataImage img, FILE **fp, char* name, long *position);
int savingChunk(DataImage img, FILE **fp, int dim, int offset);
int convolve2D(int* inbuf, int* outbuf, int sizeX, int sizeY, float* kernel, int ksizeX, int ksizeY);
void freeImagestructure(DataImage *src);

//Open Image file and image struct initialization
DataImage initimage(char* name, FILE **fp,int partitions, int halo){
    char c;
    char comment[300];
    int i=0,chunk=0;
    DataImage img=NULL;
    
    /*Se habre el fichero ppm*/

    if ((*fp=fopen(name,"r"))==NULL){
        perror("Error: ");
    }
    else{
        //Memory allocation
        img=(DataImage) malloc(sizeof(struct imagenppm));

        //Reading the first line: Magical Number "P3"
        fscanf(*fp,"%c%d ",&c,&(img->P));
        
        //Reading the image comment
        while((c=fgetc(*fp))!= '\n'){comment[i]=c;i++;}
        comment[i]='\0';
        //Allocating information for the image comment
        img->comment = calloc(strlen(comment),sizeof(char));
        strcpy(img->comment,comment);
        //Reading image dimensions and color resolution
        fscanf(*fp,"%d %d %d",&img->width,&img->height,&img->maxcolor);
        chunk = img->width*img->height / partitions;
        //We need to read an extra row.
        chunk = chunk + img->width * halo;
        if ((img->R=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->G=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->B=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    }
    return img;
}

//Duplicate the Image struct for the resulting image
DataImage duplicateImageData(DataImage src, int partitions, int halo){
    char c;
    char comment[300];
    unsigned int imageX, imageY;
    int i=0, chunk=0;
    //Struct memory allocation
    DataImage dst=(DataImage) malloc(sizeof(struct imagenppm));

    //Copying the magic number
    dst->P=src->P;
    //Copying the string comment
    dst->comment = calloc(strlen(src->comment),sizeof(char));
    strcpy(dst->comment,src->comment);
    //Copying image dimensions and color resolution
    dst->width=src->width;
    dst->height=src->height;
    dst->maxcolor=src->maxcolor;
    chunk = dst->width*dst->height / partitions;
    //We need to read an extra row.
    chunk = chunk + src->width * halo;
    if ((dst->R=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    if ((dst->G=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    if ((dst->B=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    return dst;
}

//Read the corresponding chunk from the source Image
int readImage(DataImage img, FILE **fp, int dim, int halosize, long *position){
    int i=0, k=0,haloposition=0;

    if (fseek(*fp,*position,SEEK_SET))
        perror("Error: ");
    haloposition = dim-(img->width*halosize*2);
    for(i=0;i<dim;i++) {
        // When start reading the halo store the position in the image file
        if (halosize != 0 && i == haloposition) *position=ftell(*fp);
        fscanf(*fp,"%d %d %d ",&img->R[i],&img->G[i],&img->B[i]);
        k++;
    }

    printf ("Readed = %d pixels, posicio=%lu\n",k,*position);
    return 0;
}

//Duplication of the  just readed source chunk to the destiny image struct chunk
int duplicateImageChunk(DataImage src, DataImage dst, int dim){
    int i=0;
    
    for(i=0;i<dim;i++){
        dst->R[i] = src->R[i];
        dst->G[i] = src->G[i];
        dst->B[i] = src->B[i];
    }
//    printf ("Duplicated = %d pixels\n",i);
    return 0;
}

// Open kernel file and reading kernel matrix. The kernel matrix 2D is stored in 1D format.
kernelData readKernel(char* name){
    FILE *fp;
    int i=0;
    kernelData kern=NULL;
    
    /*Opening the kernel file*/
    fp=fopen(name,"r");
    if(!fp){
        perror("Error: ");
    }
    else{
        //Memory allocation
        kern=(kernelData) malloc(sizeof(struct structkernel));
        
        //Reading kernel matrix dimensions
        fscanf(fp,"%d,%d,", &kern->kernelX, &kern->kernelY);
        kern->vkern = (float *)malloc(kern->kernelX*kern->kernelY*sizeof(float));
        
        // Reading kernel matrix values
        for (i=0;i<(kern->kernelX*kern->kernelY)-1;i++){
            fscanf(fp,"%f,",&kern->vkern[i]);
        }
        fscanf(fp,"%f",&kern->vkern[i]);
        fclose(fp);
    }
    return kern;
}

// Open the image file with the convolution results
int initfilestore(DataImage img, FILE **fp, char* name, long *position){
    /*Se crea el fichero con la imagen resultante*/
    if ( (*fp=fopen(name,"w")) == NULL ){
        perror("Error: ");
        return -1;
    }
    /*Writing Image Header*/
    fprintf(*fp,"P%d\n%s\n%d %d\n%d\n",img->P,img->comment,img->width,img->height,img->maxcolor);
    *position = ftell(*fp);
    return 0;
}

// Writing the image partition to the resulting file. dim is the exact size to write. offset is the displacement for avoid halos.
int savingChunk(DataImage img, FILE **fp, int dim, int offset){
    int i,k=0;
    //Writing image partition
    for(i=offset;i<dim+offset;i++){
        fprintf(*fp,"%d %d %d ",img->R[i],img->G[i],img->B[i]);
//        if ((i+1)%6==0) fprintf(*fp,"\n");
        k++;
    }
//    printf ("Writed = %d pixels, dim=%d, offset=%d\n",k,dim, offset);
    return 0;
}

// This function free the space allocated for the image structure.
void freeImagestructure(DataImage *src){
    
    free((*src)->comment);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);
    
    free(*src);
}

///////////////////////////////////////////////////////////////////////////////
// 2D convolution
// 2D data are usually stored in computer memory as contiguous 1D array.
// So, we are using 1D array for 2D data.
// 2D convolution assumes the kernel is center originated, which means, if
// kernel size 3 then, k[-1], k[0], k[1]. The middle of index is always 0.
// The following programming logics are somewhat complicated because of using
// pointer indexing in order to minimize the number of multiplications.
//
//
// signed integer (32bit) version:
///////////////////////////////////////////////////////////////////////////////
int convolve2D(int* in, int* out, int dataSizeX, int dataSizeY,
               float* kernel, int kernelSizeX, int kernelSizeY)
{
    
    // check validity of params
    if(!in || !out || !kernel) return -1;
    if(dataSizeX <= 0 || kernelSizeX <= 0) return -1;

    #pragma omp parallel
    {
    int *inPtr, *inPtr2, *outPtr;
    float *kPtr;
    int kCenterX, kCenterY;
    
    // find center position of kernel (half of kernel size)
    kCenterX = (int)kernelSizeX / 2;
    kCenterY = (int)kernelSizeY / 2;
    
    // start convolution

    int i, j, m, n;
    int rowMin, rowMax;                             // to check boundary of input array
    int colMin, colMax;                             //
    float sum;                                      // temp accumulation buffer

    kPtr = kernel;

    #if defined(STATIC)
    int firstTime = TRUE;
    #pragma omp for schedule(static)
    #elif defined(DYNAMIC)
    #pragma omp for schedule(dynamic)
    #elif defined(GUIDED)
    #pragma omp for schedule(guided)
    #endif
    for(i= 0; i < dataSizeY; ++i)                   // number of rows
    {
        //Pointers must start at the correct place for each thread
        //The initial i variable is the first line where every thread starts
        #if defined(STATIC)
        if(firstTime) {
            inPtr = inPtr2 = &in[i * dataSizeX + dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
            outPtr = &out[i * dataSizeX];
            firstTime=0;
        }
        #else
        //In the other cases, the pointers must be set because OMP manage
        //the ranges and intervals of the assigned rows
        inPtr = inPtr2 = &in[i * dataSizeX + dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
        outPtr = &out[i * dataSizeX];
        #endif
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;
        
        for(j = 0; j < dataSizeX; ++j)              // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;
            
            sum = 0;                                // set to 0 before accumulate
            
            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for(m = 0; m < kernelSizeY; ++m)        // kernel rows
            {
                // check if the index is out of bound of input array
                if(m <= rowMax && m > rowMin)
                {
                    for(n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if(n <= colMax && n > colMin) {
                            sum += *(inPtr - n) * *kPtr;
                        }
                        
                        ++kPtr;                     // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;            // out of bound, move to next row of kernel
                
                inPtr -= dataSizeX;                 // move input data 1 raw up
            }
            
            // convert integer number
            if(sum >= 0) *outPtr = (int)(sum + 0.5f);
//            else *outPtr = (int)(sum - 0.5f)*(-1);
            // For using with image editors like GIMP or others...
//            else *outPtr = (int)(sum - 0.5f);
            // For using with a text editor that read ppm images like libreoffice or others...
            else *outPtr = 0;
            kPtr = kernel;                          // reset kernel to (0,0)
            inPtr = ++inPtr2;                       // next input
            ++outPtr;                               // next output
        }
    }
    }
    return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int i=0,j=0,k=0;
//    int headstored=0, imagestored=0, stored;
    
    if(argc != 6)
    {
        printf("Usage: %s <image-file> <kernel-file> <result-file> <partitions> <threads>\n", argv[0]);
        
        printf("\n\nError, Missing parameters:\n");
        printf("format: ./serialconvolution image_file kernel_file result_file\n");
        printf("- image_file : source image path (*.ppm)\n");
        printf("- kernel_file: kernel path (text file with 1D kernel matrix)\n");
        printf("- result_file: result image path (*.ppm)\n");
        printf("- partitions : Image partitions\n");
        printf("- threads: number of threads for OMP\n\n");
        return -1;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // READING IMAGE HEADERS, KERNEL Matrix, DUPLICATE IMAGE DATA, OPEN RESULTING IMAGE FILE
    //////////////////////////////////////////////////////////////////////////////////////////////////
    int imagesize, partitions, partsize, chunksize, halo, halosize, width, height;
    long position=0;
    double start, tstart=0, tend=0, tread=0, tcopy=0, tconv=0, tstore=0, treadk=0;
    struct timeval tim;
    int threads;
    FILE *fpsrc=NULL,*fpdst=NULL;
    DataImage source=NULL, output=NULL;

    // Store number of partitions
    partitions = atoi(argv[4]);

    threads = atoi(argv[5]);

    omp_set_dynamic(FALSE);
    omp_set_num_threads(threads);

    ////////////////////////////////////////
    //Reading kernel matrix
    gettimeofday(&tim, NULL);
    start = tim.tv_sec+(tim.tv_usec/1000000.0);
    tstart = start;
    kernelData kern=NULL;
    if ( (kern = readKernel(argv[2]))==NULL) {
        //        free(source);
        //        free(output);
        return -1;
    }
    //The matrix kernel define the halo size to use with the image. The halo is zero when the image is not partitioned.
    if (partitions==0) {
        printf("partitions must be higher than 0.\n");
        return;
    }
    if (partitions==1) halo=0;
    else halo = (kern->kernelY/2)*2; // This operation subtract 1 if kernelY is odd
    gettimeofday(&tim, NULL);
    treadk = treadk + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);

    ////////////////////////////////////////
    //Reading Image Header. Image properties: Magical number, comment, size and color resolution.
    gettimeofday(&tim, NULL);
    start = tim.tv_sec+(tim.tv_usec/1000000.0);
    //Memory allocation based on number of partitions and halo size.
    if ( (source = initimage(argv[1], &fpsrc, partitions, halo)) == NULL) {
        return -1;
    }

    width = source->width;
    height = source->height;

    gettimeofday(&tim, NULL);
    tread = tread + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
    
    //Duplicate the image struct.
    gettimeofday(&tim, NULL);
    start = tim.tv_sec+(tim.tv_usec/1000000.0);
    if ( (output = duplicateImageData(source, partitions, halo)) == NULL) {
        return -1;
    }
    gettimeofday(&tim, NULL);
    tcopy = tcopy + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
    
    ////////////////////////////////////////
    //Initialize Image Storing file. Open the file and store the image header.
    gettimeofday(&tim, NULL);
    start = tim.tv_sec+(tim.tv_usec/1000000.0);
    if (initfilestore(output, &fpdst, argv[3], &position)!=0) {
        perror("Error: ");
        //        free(source);
        //        free(output);
        return -1;
    }
    gettimeofday(&tim, NULL);
    tstore = tstore + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // CHUNK READING
    //////////////////////////////////////////////////////////////////////////////////////////////////
    int c, offset=0;
    imagesize = source->height*source->width;
    partsize  = (source->height*source->width)/partitions;
    printf("%s ocupa %dx%d=%d pixels. Partitions=%d, halo=%d, partsize=%d pixels\n", argv[1], source->height, source->width, imagesize, partitions, halo, partsize);

    for(c=0; c<partitions; c++) {
        ////////////////////////////////////////////////////////////////////////////////
        //Reading Next chunk.
        gettimeofday(&tim, NULL);
        start = tim.tv_sec+(tim.tv_usec/1000000.0);
        if (c==0) {
            halosize  = halo/2;
            offset   = 0;
        }
        else if(c<partitions-1) {
            halosize  = halo;
            offset    = (source->width*halo/2);
        }
        else {
            halosize  = halo/2;
            offset    = (source->width*halo/2);
        }
        chunksize = partsize + (source->width*halosize);
        //DEBUG
        //printf("\nRound = %d, position = %ld, partsize= %d, chunksize=%d pixels\n", c, position, partsize, chunksize);

        if (readImage(source, &fpsrc, chunksize, halo/2, &position)) {
            return -1;
        }

        gettimeofday(&tim, NULL);
        tread = tread + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
        
        //Duplicate the image chunk
        gettimeofday(&tim, NULL);
        start = tim.tv_sec+(tim.tv_usec/1000000.0);
        if ( duplicateImageChunk(source, output, chunksize) ) {
            //return -1;
        }
        //DEBUG
        //for (i=0;i<chunksize;i++)
        //    if (source->R[i]!=output->R[i] || source->G[i]!=output->G[i] || source->B[i]!=output->B[i]) printf("At position i=%d %d!=%d,%d!=%d,%d!=%d\n",i,source->R[i],output->R[i], source->G[i],output->G[i],source->B[i],output->B[i]);
        gettimeofday(&tim, NULL);
        tcopy = tcopy + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
        
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // CHUNK CONVOLUTION
        //////////////////////////////////////////////////////////////////////////////////////////////////
        gettimeofday(&tim, NULL);
        start = tim.tv_sec+(tim.tv_usec/1000000.0);
        
        convolve2D(source->R,
                   output->R,
                   source->width,
                   (source->height/partitions)+halosize,
                   kern->vkern,
                   kern->kernelX,
                   kern->kernelY);

        convolve2D(source->G,
                   output->G,
                   source->width,
                   (source->height/partitions)+halosize,
                   kern->vkern,
                   kern->kernelX,
                   kern->kernelY);

        convolve2D(source->B,
                   output->B,
                   source->width,
                   (source->height/partitions)+halosize,
                   kern->vkern,
                   kern->kernelX,
                   kern->kernelY);
        
        gettimeofday(&tim, NULL);
        tconv = tconv + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
        
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // CHUNK SAVING
        //////////////////////////////////////////////////////////////////////////////////////////////////
        //Storing resulting image partition.
        gettimeofday(&tim, NULL);
        start = tim.tv_sec+(tim.tv_usec/1000000.0);

        if (savingChunk(output, &fpdst, partsize, offset)) {
            perror("Error: ");
            //        free(source);
            //        free(output);
            return -1;
        }
        gettimeofday(&tim, NULL);
        tstore = tstore + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
    }

    fclose(fpsrc);
    fclose(fpdst);
    
    freeImagestructure(&source);
    freeImagestructure(&output);
    
    gettimeofday(&tim, NULL);
    tend = tim.tv_sec+(tim.tv_usec/1000000.0);
    
    printf("Image  : %s\n", argv[1]);
    printf("ISizeX : %d\n", width);
    printf("ISizeY : %d\n", height);
    printf("kSizeX : %d\n", kern->kernelX);
    printf("kSizeY : %d\n", kern->kernelY);
    printf("%.6lf seconds elapsed for Reading image file.\n", tread);
    printf("%.6lf seconds elapsed for copying image structure.\n", tcopy);
    printf("%.6lf seconds elapsed for Reading kernel matrix.\n", treadk);
    printf("%.6lf seconds elapsed for make the convolution.\n", tconv);
    printf("%.6lf seconds elapsed for writing the resulting image.\n", tstore);
    printf("%.6lf seconds elapsed\n", tend-tstart);
    //<image>;<kernel>;<result>;<partitions>;<threads>;
    //<reading>;<copying>;<reading-kernel>;<convolution>;<writing>;<total>
    printf("%s;%s;%s;%s;%s;%.6lf;%.6lf;%.6lf;%.6lf;%.6lf;%.6lf\n",
        argv[1],argv[2],argv[3],argv[4],argv[5],tread,tcopy,treadk,
        tconv,tstore,tend-tstart);
    return 0;
}
