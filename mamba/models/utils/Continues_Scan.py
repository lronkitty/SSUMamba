import torch

def continues_scan(input):
    # t,h,w = input.shape
    # input = input.pad((0,1,0,1,0,1), mode='constant')
    # t_,h_,w_ = input.shape
    input_rev = input.flip(-1)
    input_ = input.clone()
    input_[:,:,:,1::2,:] = 0
    input_rev[:,:,:,0::2,:] = 0
    input_ = torch.add(input_, input_rev)
    # input_rev = input.flip(-2).contiguous()
    # input_rev[:,:,1:2] = 0
    # input[:,:,0:1] = 0
    # input = input+input_rev
    input_rev_ = input_.flip(-2).flip(-1)
    input_rev_[:,:,0::2,:,:] = 0
    input_[:,:,1::2,:,:] = 0
    # output = input+input_rev
    output = torch.add(input_, input_rev_)
    return output

def rev_continues_scan(input):
    input_rev = input.flip(-1).flip(-2)
    input_ = input.clone()
    input_rev[:,:,0::2,:,:] = 0
    input_[:,:,1::2,:,:] = 0
    input_ = torch.add(input_, input_rev)
    input_rev_ = input_.flip(-1)
    input_[:,:,:,1::2,:] = 0
    input_rev_[:,:,:,0::2,:] = 0
    output = torch.add(input_, input_rev_)
    return output

if __name__ == '__main__':
    tensor_3d = torch.as_tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],
                      
                      [[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]],
                      
                      [[19, 20, 21],
                       [22, 23, 24],
                       [25, 26, 27]]])
    # tensor_3d = torch.as_tensor([[[1, 2],
    #                               [3, 4]],
    #                              [[5, 6],
    #                               [7, 8]]])
    tensor_3d = tensor_3d.unsqueeze(0).unsqueeze(0)
    x  = continues_scan(tensor_3d)
    print(x.view(-1))
    x = rev_continues_scan(x)
    print(x.view(-1))