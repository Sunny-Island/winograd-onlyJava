

#define MAX_BATCH           128
#define MAX_IMAGE_CHANNELS  64
#define MAX_IROWS           1024
#define MAX_FILTER_CHANNELS 512
#define MAX_FILTERS         2048

#define BATCH_TOGETHER		0
#define BATCH_BLOCK			1

#define F2X3				2

#if 0
const long MAX_TILES = (MAX_IROWS-2)*(MAX_IROWS-2)*0.25; 
long ISTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 
long OSTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
#else
extern long ISTRIDE; 
extern long FSTRIDE;
extern long OSTRIDE;
#endif
extern float* t_filter;    
extern float* t_image;    
extern float* c_out;    