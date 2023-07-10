import numpy as np
import matplotlib.pyplot as plt
class Checker:
    def __init__(self,resolution,tile_size):
        # check whether 'if' condition needed here
        self.resolution=resolution
        self.tile_size=tile_size
        self.output=None
    def draw(self):
        # if self.resolution % (2*self.tile_size) == 0:
        # pat[::2, 1::2] = 1
        dx=self.resolution//self.tile_size
        pat=np.zeros((dx,dx),dtype=int)
        pat[1::2,::2]=1
        pat[::2,1::2]=1
        pat_final=np.kron(pat,np.ones((self.tile_size,self.tile_size),dtype=int))
        self.output=pat_final
        # for i in range(0,self.resolution,self.tile_size):
        #     for j in range(0,self.resolution,self.tile_size):
        #         if (i//self.tile_size)%2 == (j//self.tile_size)%2:
        #             self.output[i:(i+self.tile_size),j:(j+self.tile_size)]=0
        #         else:
        #             self.output[i:(i + self.tile_size), j:(j + self.tile_size)]=1
        return self.output.copy()
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()

class Circle:
    def __init__(self,resolution,radius,position):
        self.resolution=resolution
        self.radius=radius
        self.position=position
        self.output=None
    def draw(self):
        matrix=np.zeros((self.resolution,self.resolution))
        x_center,y_center=self.position
        x,y=np.meshgrid(range(self.resolution),range(self.resolution))
        dist=np.sqrt((x-x_center)**2 + (y-y_center)**2)
        pattern=np.where(dist<=self.radius,1,0)
        print(pattern)
        matrix += pattern
        self.output=matrix
        return self.output.copy()
    def show(self):
        plt.imshow(self.output,cmap='gray')
        plt.axis('off')
        plt.show()
class Spectrum:
    def __init__(self,resolution):
        self.resolution=resolution
        self.output=None
    def draw(self):
        mat=np.zeros([self.resolution,self.resolution,3]) #3D array
        r_comp=np.linspace(1,0,self.resolution)
        mat[:,:,2]=r_comp
        g_comp = np.linspace(0,1,self.resolution).reshape(self.resolution,1)
        mat[:,:,1] = g_comp
        b_comp=np.linspace(0,1,self.resolution)
        mat[:,:,0]=b_comp
        self.output=mat
        return self.output.copy()
    def show(self):
        plt.imshow(self.output)
        plt.axis('off')
        plt.show()