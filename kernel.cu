#include <stdint>
#include "interval.cuh"

#define SURFACE 0.f
#define COUNT_OCCUPIED false
//radius of finite difference stencil
#define RAD 1

#define EPS 1e-7

#define MAX_DEPTH 3
//length of buffer that stores quadrature contributions
#define QUAD_LENGTH 10

#define LOG2DIM_INTERNAL_BRICK 5u
#define INTERNAL_BRICK (1u << LOG2DIM_INTERNAL_BRICK)
#define INTERNAL_BRICK_TOTAL (INTERNAL_BRICK*INTERNAL_BRICK*INTERNAL_BRICK)
#define INTERNAL_THREADS 128
#define INTERNAL_BLOCKS 256

#define LOG2DIM_LEAF_BRICK 5u
#define LEAF_BRICK (1u << LOG2DIM_LEAF_BRICK)
#define LEAF_BRICK_TOTAL (LEAF_BRICK*LEAF_BRICK*LEAF_BRICK)
#define LEAF_THREADS 1024
#define LEAF_BLOCKS 4

/////////////////////////////////////////////
//defining function (f)
/////////////////////////////////////////////

template<typename I>
__device__ inline I sphere(I x, I y, I z){
	float r = 2;

	// return sqrt(x*x + y*y + z*z) - r;
	return x*x + y*y + z*z - r*r;
}

template<typename I>
__device__ inline I super_ellipse(I x, I y, I z){
	float r = 2;

	return x*x*x*x + y*y*y*y + z*z*z*z - r*r*r*r;
}

template<typename I>
__device__ inline I cube(I x, I y, I z){
	float l = 2;

	return max(max(abs(x), abs(y)), abs(z)) - l;
}

template<typename I>
__device__ inline I dense(I x, I y, I z){
	return min(abs(x), 0.f);
}

template<typename I>
__device__ inline I f(I x, I y, I z){

	return sphere<I>(x,y,z);

	// return super_ellipse<I>(x,y,z);

	// return cube<I>(x,y,z);

	// return dense<I>(x,y,z);
}

/////////////////////////////////////////////
//integrand (g)
/////////////////////////////////////////////

__device__ inline float g(float x, float y, float z){
	return 1;
}







__device__ inline float3 __fadd_rd(float3 x, float3 y){
    return make_float3(__fadd_rd(x.x, y.x), __fadd_rd(x.y, y.y), __fadd_rd(x.z, y.z));
}
__device__ inline float3 __fadd_ru(float3 x, float3 y){
    return make_float3(__fadd_ru(x.x, y.x), __fadd_ru(x.y, y.y), __fadd_ru(x.z, y.z));
}

__device__ inline float3 __fadd_rd(float3 x, float y){
    return make_float3(__fadd_rd(x.x, y), __fadd_rd(x.y, y), __fadd_rd(x.z, y));
}
__device__ inline float3 __fadd_ru(float3 x, float y){
    return make_float3(__fadd_ru(x.x, y), __fadd_ru(x.y, y), __fadd_ru(x.z, y));
}

__device__ inline float3 __fmul_rd(float3 x, float y){
    return make_float3(__fmul_rd(x.x, y), __fmul_rd(x.y, y), __fmul_rd(x.z, y));
}
__device__ inline float3 __fmul_ru(float3 x, float y){
    return make_float3(__fmul_ru(x.x, y), __fmul_ru(x.y, y), __fmul_ru(x.z, y));
}

__device__ inline float warp_sum_reduce(float val){
	static const uint32_t mask = 0xffffffff;

	int offset = warpSize >> 1;
	while (offset > 0){
		//CUDA shfl_down_sync only supports ints so float must be cast to int and back 
		int* ival = reinterpret_cast<int*>(&val);
		int ishfl = __shfl_down_sync(mask, ival[0], offset);
		float shfl = reinterpret_cast<float*>(&ishfl)[0];
		val += shfl;
		offset >>= 1;
	}
	return val;
}

__device__ inline float block_sum_reduce(float val, float* shared){
	__syncthreads();

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	//each warp within block computes its warp sum
	val = warp_sum_reduce(val);

	//first thread of each warp writes its warp sum back into shared memory (1024 max threads per block / 32 threads per warp = 32 indices of shared memory that are written to)
	if (lane == 0) shared[warp] = val;

	__syncthreads();

	//threads in first warp load in 32 values of shared memory written to from previous step
	val = (threadIdx.x < (blockDim.x + warpSize - 1)/warpSize) ? shared[lane] : 0;

	//perform final warp shuffle on first warp to get total sum for the block
	if (warp == 0) val = warp_sum_reduce(val);

	return val;
}

__device__ inline bool chi(float f){
	return f <= SURFACE;
}

__device__ inline int3 chi(float3 v){
	return make_int3(chi(v.x), chi(v.y), chi(v.z));
}

template<int rad>
__device__ inline void central_difference(float* result, float3 r, float h){
	float3 p1 = make_float3(f<float>(r.x + h,r.y,r.z), f<float>(r.x,r.y + h,r.z), f<float>(r.x,r.y,r.z + h));
	float3 m1  = make_float3(f<float>(r.x - h,r.y,r.z), f<float>(r.x,r.y - h,r.z), f<float>(r.x,r.y,r.z - h));

	float3 df = p1 - m1;
	float3 dchi = make_float3(chi(p1) - chi(m1));

	float num = sum(df*dchi);
	float den = sum(df*df);

	if (den > EPS){
		*result -= ldexpf((h*h)*g(r.x,r.y,r.z)*num*rsqrtf(den), -1);
	}
}

template<>
__device__ inline void central_difference<2>(float* result, float3 r, float h){
	float3 p2 = make_float3(f<float>(r.x + 2*h,r.y,r.z), f<float>(r.x,r.y + 2*h,r.z), f<float>(r.x,r.y,r.z + 2*h));
	float3 p1 = make_float3(f<float>(r.x +   h,r.y,r.z), f<float>(r.x,r.y +   h,r.z), f<float>(r.x,r.y,r.z +   h));
	float3 m1 = make_float3(f<float>(r.x -   h,r.y,r.z), f<float>(r.x,r.y -   h,r.z), f<float>(r.x,r.y,r.z -   h));
	float3 m2 = make_float3(f<float>(r.x - 2*h,r.y,r.z), f<float>(r.x,r.y - 2*h,r.z), f<float>(r.x,r.y,r.z - 2*h));

	float3 df = (-p2 + 8.f*p1 - 8.f*m1 + m2)/12.f;
	float3 dchi = make_float3(-chi(p2) + 8*chi(p1) - 8*chi(m1) + chi(m2))/12.f;

	float num = sum(df*dchi);
	float den = sum(df*df);

	if (den > EPS){
		*result -= (h*h)*g(r.x,r.y,r.z)*num*rsqrtf(den);
	}
}

__device__ inline bool split_levelset(const Interval& interval, float levelset){
	return interval.lo() <= levelset && levelset <= interval.hi();
}

__device__ inline uint32_t determine_span(uint8_t depth){
	uint32_t span = LOG2DIM_LEAF_BRICK;
	for (uint8_t n = 0; n < depth; ++n)
		span += LOG2DIM_INTERNAL_BRICK;
	return 1u << span;
}

__device__ inline uint3 determine_shape(float3 box, float h){
	return make_uint3(ceilf((box)/h - 1e-5)) + 1;
}

__device__ inline uint8_t determine_depth(uint3 shape){
	uint8_t depth = 0;
	shape = shape >> LOG2DIM_LEAF_BRICK;
	for (; depth < MAX_DEPTH; ++depth){
		if (shape.x + shape.y + shape.z == 0)
			break;
		shape = shape >> LOG2DIM_INTERNAL_BRICK;
	}
	//make sure there is always at least one level within tree
	depth = depth == 0 ? 1 : depth;
	return depth;
}

__device__ inline void threadBlockDeviceSynchronize(void) {
	__syncthreads();
	if(threadIdx.x == 0)
		cudaDeviceSynchronize();
}

__global__ void internal(float* quad, uint64_t* occupancy, uint3* shape, uint3 origin, float3* offset, float h, uint8_t depth);
__global__ void leaf(float* quad, uint64_t* occupancy, uint3* shape, uint3 origin, float3* offset, float h);

__device__ inline void subdivide(float* quad, uint64_t* occupancy, uint3* shape, uint3 origin, float3* offset, uint32_t span, float h, uint8_t depth){
	//need to subtract RAD as float from lo because shape is a uint that can overflow if origin - RAD is less than 0
	float3 lo = __fadd_rd(__fadd_rd(__fmul_rd(make_float3(origin), h), __fmul_rd(RAD, -h)), *offset);
	float3 hi = __fadd_rd(__fmul_ru(make_float3(origin + span + RAD), h), *offset);
	Interval fhat = f<Interval>(Interval(lo.x, hi.x), Interval(lo.y, hi.y), Interval(lo.z, hi.z));
	if (split_levelset(fhat, SURFACE)){
		if (depth - 1 > 0){
			internal<<<INTERNAL_BLOCKS, INTERNAL_THREADS>>>(quad, occupancy, shape, origin, offset, h, depth - 1);							
		}
		else{
			leaf<<<LEAF_BLOCKS, LEAF_THREADS>>>(quad, occupancy, shape, origin, offset, h);
		}
	}
}

__global__ void leaf(float* quad, uint64_t* occupancy, uint3* shape, uint3 origin, float3* offset, float h){
	__shared__ float shared[1024];
	shared[threadIdx.x] = 0;

	float result = 0;
	for (int n = blockIdx.x*blockDim.x + threadIdx.x; n < LEAF_BRICK_TOTAL; n += blockDim.x*gridDim.x){
		uint3 leaf_origin;
		leaf_origin.x = (n >> (LOG2DIM_LEAF_BRICK + LOG2DIM_LEAF_BRICK)) + origin.x;
		leaf_origin.y = ((n >> LOG2DIM_LEAF_BRICK)%LEAF_BRICK) + origin.y;
		leaf_origin.z = (n%LEAF_BRICK) + origin.z;

		if (leaf_origin.x <= shape->x && leaf_origin.y <= shape->y && leaf_origin.z <= shape->z){
			float3 r = make_float3(leaf_origin)*h + *offset;
			central_difference<RAD>(&result, r, h);
		}
	}
	result = block_sum_reduce(result, shared);

	if (threadIdx.x == 0){
		//find first entry in `quad` that `result` can be added to without round-off
		int n = 0;
		for (; n < QUAD_LENGTH; ++n){
			if (abs(result/quad[n]) > EPS){
				break;
			}
		}
		atomicAdd(&quad[n], result);
		if (COUNT_OCCUPIED)
			atomicAdd(occupancy, 1u);
	}
}

__global__ void internal(float* quad, uint64_t* occupancy, uint3* shape, uint3 origin, float3* offset, float h, uint8_t depth){
	uint32_t span = determine_span(depth - 1);

	for (int n = blockIdx.x*blockDim.x + threadIdx.x; n < INTERNAL_BRICK_TOTAL; n += blockDim.x*gridDim.x){
		uint3 child_origin;
		child_origin.x = (n >> (LOG2DIM_INTERNAL_BRICK + LOG2DIM_INTERNAL_BRICK))*span + origin.x;
		child_origin.y = ((n >> LOG2DIM_INTERNAL_BRICK)%INTERNAL_BRICK)*span + origin.y;
		child_origin.z = (n%INTERNAL_BRICK)*span + origin.z;

		if (child_origin.x <= shape->x && child_origin.y <= shape->y && child_origin.z <= shape->z){
			subdivide(quad, occupancy, shape, child_origin, offset, span, h, depth);
		}
		//sync blocks and wait for child kernels to finish to prevent kernel queue from being exceeded
		threadBlockDeviceSynchronize();
	}
}

//to pass a pointer to variable created in a kernel to a child kernel it must be declared globally
uint3 shape;
//extern "C" is used so that the named symbols (root and dense) can be found after compilation of the code (it prevents their names from being mangled)
extern "C"{
	//kernel launch parameters: <<<1,1>>>
	__global__ void root(float* quad, uint64_t* occupancy, float3* box, float h, float3* offset){
		shape = determine_shape(*box, h);
		uint8_t depth = determine_depth(shape);
		uint32_t span = determine_span(depth - 1);
		uint3 children = (shape + span - 1)/span;

		for (uint32_t i = 0; i < children.x; ++i){
			for (uint32_t j = 0; j < children.y; ++j){
				for (uint32_t k = 0; k < children.z; ++k){
					uint3 child_origin = make_uint3(i,j,k)*span;
					// printf("%u %u %u\n", i,j,k);
					subdivide(quad, occupancy, &shape, child_origin, offset, span, h, depth);
					cudaDeviceSynchronize();
				}
			}
		}
	}

	//kernel launch parameters: <<<N,1024>>>
	__global__ void dense(float* quad, uint32_t* box, float3* offset, float h){
		__shared__ float shared[1024];
		shared[threadIdx.x] = 0;

		ulonglong3 shape = make_ulonglong3(box[0], box[1], box[2]);

		float result = 0;
		for (uint64_t n = blockIdx.x*blockDim.x + threadIdx.x; n < shape.x*shape.y*shape.z; n += blockDim.x*gridDim.x){
			uint3 origin;
			origin.x = (n/(box[2]*box[1]));
			origin.y = (n/box[2])%box[1];
			origin.z = n%box[2];

			float3 r = make_float3(origin)*h + *offset;

			float3 right = make_float3(f<float>(r.x + h,r.y,r.z), f<float>(r.x,r.y + h,r.z), f<float>(r.x,r.y,r.z + h));
			float3 left  = make_float3(f<float>(r.x - h,r.y,r.z), f<float>(r.x,r.y - h,r.z), f<float>(r.x,r.y,r.z - h));

			float3 df = right - left;
			float3 dchi = make_float3(chi(right) - chi(left));

			float num = sum(df*dchi);
			float den = sum(df*df);

			if (den > EPS){
				result -= ldexpf((h*h)*g(r.x,r.y,r.z)*num*rsqrtf(den), -1);
			}
		}
		result = block_sum_reduce(result, shared);
		if (threadIdx.x == 0) atomicAdd(quad, result);
	}
}
