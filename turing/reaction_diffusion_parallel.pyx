#!python
#cython: wraparound=False
#cython: boundscheck=False


import numpy  as np
cimport numpy as np
import cython

from cython.parallel import prange, parallel

 
def reaction_diffusion_2d(double[:,:,:] arr, double[:,:] laplace,
	double diffa, double diffb, double feed, double kill, int niter=2**10):

#	cdef int i

	for _ in range(niter):
		reaction_diffusion_2d_iter(arr, laplace, diffa, diffb, feed, kill)
#		if not i % 32: print('iter', i)



cdef reaction_diffusion_2d_iter(double[:,:,:] arr, double[:,:] laplace,
	double diffa, double diffb, double feed, double kill):

	cdef int i,j

	cdef double[:,:] arr0 = np.zeros_like(arr[0,:,:])
	cdef double[:,:] arr1 = np.zeros_like(arr[1,:,:])

	diffusion(arr0, arr[0,:,:], laplace, diffa)
	reaction_a(arr0, arr, feed)

	diffusion(arr1, arr[1,:,:], laplace, diffb)
	reaction_b(arr1, arr, feed, kill)

	for i in prange(arr.shape[1],nogil=True):
		for j in range(arr.shape[2]):
			arr[0,i,j] += arr0[i,j]
			arr[1,i,j] += arr1[i,j]


cdef reaction_a(double[:,:] arr, double[:,:,:] AB, double feed):
	cdef int i,j
	cdef double val

	for i in prange(arr.shape[0],nogil=True):
		for j in range(arr.shape[1]):
			val = feed*(1.0 - AB[0,i,j]) - AB[0,i,j]*AB[1,i,j]*AB[1,i,j]
			arr[i,j] = arr[i,j] + val

cdef reaction_b(double[:,:] arr, double[:,:,:] AB, double feed, double kill):
	cdef int i,j
	cdef double val
	cdef double fkill = feed + kill

	cdef Py_ssize_t N=arr.shape[0]

	for i in prange(arr.shape[0],nogil=True):
		for j in range(arr.shape[1]):
			val = AB[0,i,j]*AB[1,i,j]*AB[1,i,j] - fkill*AB[1,i,j]
			arr[i,j] = arr[i,j] + val



cdef diffusion(double[:,:] B, double[:,:] A, double[:,:] L, double diff):
	cdef int N = A.shape[0]
	cdef int P = A.shape[1]
	cdef int i,j

	cdef double v

	# top left
	v =  L[0,0]*A[N-1,P-1] + L[0,1]*A[N-1,0] + L[0,2]*A[N-1,1]
	v += L[1,0]*A[0,  P-1] + L[1,1]*A[0,  0] + L[1,2]*A[0,  1]
	v += L[2,0]*A[1,  P-1] + L[2,1]*A[1,  0] + L[2,2]*A[1,  1]
	B[0,0] = diff*v

	# top right
	v =  L[0,0]*A[N-1,P-2] + L[0,1]*A[N-1,P-1] + L[0,2]*A[N-1,0]
	v += L[1,0]*A[0,  P-2] + L[1,1]*A[0,  P-1] + L[1,2]*A[0,0]
	v += L[2,0]*A[1,  P-2] + L[2,1]*A[1,  P-1] + L[2,2]*A[1,0]
	B[0,P-1] = diff*v

	# bottom left
	v =  L[0,0]*A[N-2,P-1] + L[0,1]*A[N-2,0] + L[0,2]*A[N-2,1]
	v += L[1,0]*A[N-1,P-1] + L[1,1]*A[N-1,0] + L[1,2]*A[N-1,1]
	v += L[2,0]*A[0,  P-1] + L[2,1]*A[0,0]   + L[2,2]*A[0,1]
	B[N-1,0] = diff*v
	
	# bottom right
	v =  L[0,0]*A[N-2,P-2] + L[0,1]*A[N-2,P-1] + L[0,2]*A[N-2,0]
	v += L[1,0]*A[N-1,P-2] + L[1,1]*A[N-1,P-1] + L[1,2]*A[N-1,0]
	v += L[2,0]*A[0,  P-2] + L[2,1]*A[0,  P-1] + L[2,2]*A[0,  0]
	B[N-1,P-1] = diff*v


	# top row
	for j in prange(1,P-1,nogil=True):
		v =  L[0,0]*A[N-1,j-1] + L[0,1]*A[N-1,j] + L[0,2]*A[N-1,j+1]
		v = v + L[1,0]*A[0,  j-1] + L[1,1]*A[0,  j] + L[1,2]*A[0,  j+1]
		v = v + L[2,0]*A[1,  j-1] + L[2,1]*A[1,  j] + L[2,2]*A[1,  j+1]
		B[0,j] = diff*v

	# bottom row
	for j in prange(1,P-1,nogil=True):
		v =  L[0,0]*A[N-2,j-1] + L[0,1]*A[N-2,j] + L[0,2]*A[N-2,j+1]
		v = v + L[1,0]*A[N-1,j-1] + L[1,1]*A[N-1,j] + L[1,2]*A[N-1,j+1]
		v = v + L[2,0]*A[0,  j-1] + L[2,1]*A[0,  j] + L[2,2]*A[0,  j+1]
		B[N-1,j] = diff*v

	# left column
	for i in prange(1,N-1,nogil=True):
		v =  L[0,0]*A[i-1,P-1] + L[0,1]*A[i-1,0] + L[0,2]*A[i-1,1]
		v = v + L[1,0]*A[i,  P-1] + L[1,1]*A[i,  0] + L[1,2]*A[i,  1]
		v = v + L[2,0]*A[i+1,P-1] + L[2,1]*A[i+1,0] + L[2,2]*A[i+1,1]
		B[i,0] = diff*v

	# right column
	for i in prange(1,N-1,nogil=True):
		v =  L[0,0]*A[i-1,P-2] + L[0,1]*A[i-1,P-1] + L[0,2]*A[i-1,0]
		v = v + L[1,0]*A[i,  P-2] + L[1,1]*A[i,  P-1] + L[1,2]*A[i,  0]
		v = v + L[2,0]*A[i+1,P-2] + L[2,1]*A[i+1,P-1] + L[2,2]*A[i+1,0]
		B[i,P-1] = diff*v

	# middle
	for i in prange(1,N-1,1,nogil=True):
		for j in range(1,P-1):
			v =  L[0,0]*A[i-1,j-1] + L[0,1]*A[i-1,j] + L[0,2]*A[i-1,j+1]
			v = v + L[1,0]*A[i,  j-1] + L[1,1]*A[i,  j] + L[1,2]*A[i,  j+1]
			v = v + L[2,0]*A[i+1,j-1] + L[2,1]*A[i+1,j] + L[2,2]*A[i+1,j+1]
			B[i,j] = diff*v

