import numpy as np
import cv2

from enum import Enum
Kernelmethod = Enum('Type','gaussian exponential')

import BasicAnnotationPreprocessing




def gaussian_kernel(filter_size=[51,51], sigma=8, mean=1):
    kx = cv2.getGaussianKernel(filter_size[0], sigma)
    ky = cv2.getGaussianKernel(filter_size[1], sigma)
    k = kx * np.transpose(ky)
    k *= (mean / np.max(k))
    return k.astype(np.float32)

def exponential_Kernel(filter_size=[51,51], sigma=8,mean=1):
    center = [filter_size[0]//2,filter_size[1]//2]
    k = np.exp(-np.sqrt((Y-center[0])**2+(X-center[1])**2)/sigma)
    k *= (mean / np.max(k))
    return k.astype(np.float32)

def getCellNeighbourCoords(yPt,xPt,kernelSize,imgshape):
    deltaSize = kernelSize//2
    yCoord = [np.max([0,yPt-deltaSize]),np.min([imgshape[1],yPt+deltaSize+1])]
    xCoord = [np.max([0,xPt-deltaSize]),np.min([imgshape[0],xPt+deltaSize+1])]
    ykernel = [np.max([0,deltaSize-yPt]), np.min([imgshape[1]-yCoord[0],kernelSize])]
    xkernel = [np.max([0,deltaSize-xPt]), np.min([imgshape[0]-xCoord[0],kernelSize])]
    return xCoord, yCoord, xkernel, ykernel



class BlurKernelPreprocessing(BasicAnnotationPreprocessing.BasicAnnotationPreprocessing):
    """ Annotation preprocessor if object positions are represented by a black iamge with white pixels where objects are located
    """

    def __init__(self,kerneltype=Kernelmethod.gaussian,kernelsize=[51,51], sigma=8, parameters=None):
        super().__init__(str(type(self).__name__))
		if (parameters is None):
            self.kerneltype = kerneltype
			self.parameters['kerneltype'] = kerneltype._name_
			self.parameters['kernelsize'] = kernelsize
            self.parameters['sigma'] = sigma
		else:
			self.parameters = parameters
            self.kerneltype = Kernelmethod[parameters['type']]



    def preprocessAnnotation(self, mask):
        if self.kerneltype == Kernelmethod.gaussian:
            self.kernel = gaussian_kernel(self.parameters['kernelsize'], self.parameters['sigma'],)
        elif self.kerneltype == Kernelmethod.exponential:
            self.kernel = exponential_Kernel(self.parameters['kernelsize'], self.parameters['sigma'])
        else:
			raise TypeError('kernel type not implemented')

        probabilityMap = self.computeProbabilityMapFromPixelPosImage(mask)
        return probabilityMap


    def computeProbabilityMapFromPixelPosImage(self,posPixelImg):
        x,y = np.where(posPixelImg==255)
    
        probabilityMap = np.zeros(posPixelImg.shape,dtype=np.float32)
        kernelSize = self.kernel.shape[0]
    
    
        for i in range(0,len(x)):
            xcoord,ycoord,xkernel,ykernel = getCellNeighbourCoords(y[i],x[i],kernelSize,probabilityMap.shape)
            part = probabilityMap[xcoord[0]:xcoord[1],ycoord[0]:ycoord[1]]
            comp = np.zeros([2,part.shape[0],part.shape[1] ],dtype=np.float32)
            comp[0,:,:] = part
            comp[1,:,:] = self.kernel[xkernel[0]:xkernel[1],ykernel[0]:ykernel[1]]
            probabilityMap[xcoord[0]:xcoord[1],ycoord[0]:ycoord[1]] = np.max(comp,axis=0)
        
        return probabilityMap





    
    
    
