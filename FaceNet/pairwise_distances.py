def pairwise_distances(x, y):
  '''                                                                                              
  Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
  Input: x is a bxNxd matrix                                                                       
         y is an optional bxMxd matirx                                                             
  Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
  i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
  '''                                                                                              
  x_norm = x.norm(dim=2)[:,:,None]                                                                 
  y_t = y.permute(0,2,1).contiguous()                                                              
  y_norm = y.norm(dim=2)[:,None]                                                                   

  dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)                                                 

  return torch.clamp(dist, 0.0, np.inf)
