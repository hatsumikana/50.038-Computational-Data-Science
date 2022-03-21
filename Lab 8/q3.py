import torch
import time
import numpy

input = torch.randn(32,100,300)
seq_len = input.size()[1]
output_size = input.size()[2]
batch_size = input.size()[0]

seq_lengths = numpy.random.randint(1,seq_len + 1,batch_size)

# >>> input
# tensor([[[ 0.5600,  1.5497, -1.1143,  0.5364, -1.2418],
#          [-0.9693,  0.5391, -0.2649, -1.2627, -0.9124],
#          [-1.0370, -1.2200,  1.4667, -0.1905,  0.1095],
#          [-0.4087,  0.4417, -0.6743, -0.2452, -0.5837]],

#         [[-0.1090,  0.5922,  0.5269, -0.6568, -1.4115],
#          [ 0.7329,  0.2175,  1.4582,  1.1836,  1.2953],
#          [ 0.7877, -0.0282,  1.0278,  1.5513, -0.9504],
#          [-0.3102, -0.0876, -0.2577,  1.0907, -0.7906]],

#         [[-0.2297,  1.1952, -0.2008, -0.4737, -2.2075],
#          [-1.3182,  1.2937, -1.3283,  1.0359, -2.0254],
#          [ 0.7924, -1.2665, -0.7068,  0.3463, -0.0368],
#          [-0.3447, -0.1287,  0.9366,  1.4536,  0.3898]]])


# Approach 1: 

# Create 3D mask during preprocessing for each sample and pass to the dataloader and the forward function.
# In every 3D mask only the element whose location = length of the sample - 1 will be set to 1 and rest of the elements
# in that mask should be set 0.
# The shape of the tensor should be batch_size, seq_len, output_size. So for the sentence ``I am SJ'', considering seq_len is 4,
# and the output size is 3, the mask will be (000, 000, 111, 000 ). Similarly, for ``okay sure'', it will be
# (000, 111, 000, 000). After that, multiply this mask with the lstm output to get a 3D tensor where only the
#locations = the length of the each sample in the batch will be 1 for each output_size and rest will be 0. To make a 2D so that
#you can apply FC on it use torch.sum(mask*lstm_out, dim=1). The final output will be 2D tensor of (batch_size, output_size)

# Approach 1: Optimization

# Create the masks before creating the batches by modifying train_dataset and test_dataset objects. Then inside collate_batch
# function, create batches that contrain the mask for each sample. Afterward, pass it to the forward function. 
# This ensures that for each sample you only create mask once and reuse it in different batches. Creating mask
# inside the forward function is a bad practice as it slows down the training.


last_indices = torch.LongTensor(seq_lengths - 1)

mask = torch.zeros(batch_size, seq_len, 1)

rows = torch.arange(0, batch_size).long()
mask[rows, last_indices, :] = 1

start = time.time()

mask = mask.expand(-1,seq_len,output_size)
output = input * mask

last_states = torch.sum(output, dim=1)

end = time.time()

print("elapsed time ", end-start)
print("last states", last_states)

# Approach 2: Use PyTorch torch.select_index

# Flatten the input


input_new = input.view(-1)

# Write the formula for the indices. Suppose for the first element in the batch, the length is 3. For the second element in the batch, the length is 1. For the third element in the batch, the length is 4.

# Use this formula

# indicies = torch.tensor([0*4*5 + 5*(3-1) , 0*4*5 + 5*(3-1) + 1, 0*4*5 + 5*(3-1) + 2, 0*4*5 + 5*(3-1) + 3, 0*4*5 + 5*(3-1) + 4, 1*4*5 + 5*(1-1),  1*4*5 + 5*(1-1) + 1,  1*4*5 + 5*(1-1) + 2,  1*4*5 + 5*(1-1) + 3,  1*4*5 + 5*(1-1) + 4, 2*4*5 + 5*(4-1), 2*4*5 + 5*(4-1) +1, 2*4*5 + 5*(4-1) +2, 2*4*5 + 5*(4-1) + 3, 2*4*5 + 5*(4-1) + 4])

# To generalize, use the following code

indices = []

start = time.time()

for i,j in enumerate(input):
	for k in range(0,output_size):
		indices.append(i*seq_len*output_size + output_size * (seq_lengths[i]-1) + k)

# Convert indices to tensor

indices = torch.IntTensor(indices)

x = torch.index_select(input_new, 0, indices)

# Reshaping x to (batch_size, output_size)
x = x.view(-1, output_size)

end = time.time()

# >>> x

# tensor([[-1.0370, -1.2200,  1.4667, -0.1905,  0.1095],
#         [-0.1090,  0.5922,  0.5269, -0.6568, -1.4115],
#         [-0.3447, -0.1287,  0.9366,  1.4536,  0.3898]])

print("elapsed time ", end-start)
print("last states", x)

# Approach 3:

# Approach 3 is an optimized version of approach 2: Create indices in Approach 2 during the preprocessing.
# This can be done by modifying the train_dataset and test_dataset objects. For each sample in those datasets, add
# the indices that you want to extract. So your dataset has now text, label, indices. Afterward, inside collate_batch,
# create a indices tensor for that batch by just appending indices of each sample in a batch to a list and convert that to an IntTensor.
# The rest should be similar to the approach 2.
# As you are just creating indices of each sample one time during preprocessing, this approach is much faster than approach 2.

# But this is a bad solution as every time you change the output_size you have to create new indices 

# Approach 4:

last_indices = torch.LongTensor(seq_lengths - 1)
start = time.time()
last_indices_mod = last_indices.view(-1, 1, 1).expand(-1, 1, output_size)

last_states = input.gather(1, last_indices_mod)
end= time.time()

print("elapsed time ", end-start)
print("last states", last_states)


# Approach 5:

start = time.time()
rows = torch.arange(0, batch_size).long()
last_states = input[rows, last_indices, :]
end = time.time()

print("elapsed time ", end-start)
print("last states", last_states)
