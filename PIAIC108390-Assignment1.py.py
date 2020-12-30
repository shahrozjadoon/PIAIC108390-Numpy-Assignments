#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[5]:


import numpy as np


# 2. Create a null vector of size 10 

# In[3]:


import numpy as np
x = np.zeros(10)
print(x)


# 3. Create a vector with values ranging from 10 to 49

# In[2]:


import numpy as np
vector = np.arange(10,49)
print(vector)


# 4. Find the shape of previous array in question 3

# In[3]:


import numpy as np
l=[10, 11, 12 ,13, 14, 15, 16, 17, 18 ,19, 20, 21, 22 ,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
x=np.array(l)
np.reshape


# 5. Print the type of the previous array in question 3

# In[6]:


import numpy as np
x.dtype


# 6. Print the numpy version and the configuration
# 

# In[7]:


import numpy as np
print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[8]:


import numpy as np
a = np.array([10, 11, 12 ,13, 14, 15, 16, 17, 18 ,19, 20, 21, 22 ,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48])
print(a)
np.ndim(a)


# 8. Create a boolean array with all the True values

# In[9]:


import numpy as np
l=[10, 11, 12 ,13, 14, 15, 16, 17, 18 ,19, 20, 21, 22 ,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
x=np.array(l)
x>9


# 9. Create a two dimensional array
# 
# 
# 

# In[10]:


import numpy as np
x=np.zeros((10,10))
x.shape


# 10. Create a three dimensional array
# 
# 

# In[11]:


import numpy as np
z=x.reshape(2,2,25)
print(z)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[13]:


import numpy as np
vector = (49,10)
print(vector)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[14]:


import numpy as np
x = np.zeros(10)
print(x)
x[4] = 1
print(x)


# 13. Create a 3x3 identity matrix

# In[15]:


import numpy as np
array_2D=np.identity(3)
print(array_2D)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[ ]:


import numpy as np
arr = np.array([1, 2, 3, 4, 5]) 
arr = arr.astype('float64') 
print(arr) 
print(arr.dtype)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[ ]:


arr1 = np.array([[1., 2., 3.],
            [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],
           [7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[16]:


arr1 = np.array([[1., 2., 3.],
            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.], 
            [7., 2., 12.]])

comparison = arr1 == arr2
equal_arrays = comparison.all() 
  
print(equal_arrays)


# 17. Extract all odd numbers from arr with values(0-9)

# In[17]:


a = np.array([0,1,2,3,4,5,6,7,8,9])
a[a % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[18]:


a = np.array([0,1,2,3,4,5,6,7,8,9])
-a[a % 2 == 1]


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[19]:


import numpy as np
arr=np.arange(10)
arr[5] 
arr[5:9] 
arr[5:9] = 12
print(arr)


# 20. Create a 2d array with 1 on the border and 0 inside

# In[20]:


import numpy as np
x = np.ones((5,5))
x[1:-1,1:-1] = 0
print(x)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[ ]:


arr2d = np.array([[1, 2, 3],
            [4, 5, 6], 

            [7, 8, 9]])
arr2d[5] = 12
print(arr2d)


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[21]:


import numpy as np
arr3d = np.array([[[1, 2, 3], 
                   [4, 5, 6]], 
                  [[7, 8, 9], 
                   [10, 11, 12]]])
arr3d = arr.astype('float64') 
print(arr3d)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[22]:


x=[[0,1,2,3,4],[5,6,7,8,9]]
d=np.array(x)
np.split(d,[5])


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[51]:


x = np.arange(0,10,1).reshape(2,5)
x = x[1:,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[41]:


import numpy as np
arr=np.arange(0,10,1).reshape(2,5)
x=arr[:,2]
print(arr)
print(x)


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[24]:


import numpy as np
x = np.random.random((10,10))
xmin, xmax = x.min(), x.max()
print(xmin, xmax)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[25]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a, b))


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[ ]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b))


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[ ]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(names)
print(data)
data[names == 'Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[ ]:





# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[26]:


import numpy as np
arr = np.arange(0,15)
x = arr.reshape((5,3))
print(x)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[27]:


import numpy as np
arr = np.arange(0,16)
x = arr.reshape((2,2,4))
print(x)


# 33. Swap axes of the array you created in Question 32

# In[28]:


import numpy as np
arr = np.arange(0,16)
x = arr.reshape((2,2,4))
arr = x.swapaxes(1, 2)
print(arr)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[29]:


import numpy as np
arr = np.arange(10)
print(arr)
z = np.sqrt(arr ** 2)
print(z)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[30]:


h=np.random.rand(12)
print(h)
k=np.random.rand(12)
print(k)
hmax, kmax = h.max(), k.max()
print(hmax, kmax)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[31]:


names=np.unique(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[47]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
result = np.setdiff1d(a, b)
print(result)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[48]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
sampleArray[:,1] = newColumn


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[54]:


x = np.array([[1,2,3],[4,5,6]])
y = np.array([[6,23],[-1,7],[8,9]])
z = np.dot(x,y)
print(z)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[4]:


import numpy as np
c=np.random.rand(2,3,4)
np.cumsum(c)
c


# In[ ]:




