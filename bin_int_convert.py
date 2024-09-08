import torch
import torch.nn as nn


def int_to_signMagnitude(weight_q, w_bitwidth: int=8, device='cpu'):
    weight_q = weight_q.to(device, copy=True)
    is_min = weight_q.eq(-2**(w_bitwidth-1))
    weight_q[is_min] = -2**(w_bitwidth-1) + 1
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = -weight_q[is_neg]
    
    weight_q_shape = list(weight_q.size())
    remainder_list = torch.zeros([w_bitwidth] + weight_q_shape, device=device)

    for k in reversed(range(w_bitwidth)):
        remainder = torch.fmod(weight_q, 2)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/2)

    remainder_list[0, is_neg] = 1.
    return remainder_list


def signMagnitude_to_int(wqb_list, w_bitwidth: int=8, device='cpu'):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape, device=device)

    for k in reversed(range(int(w_bitwidth))):
        if k != 0:
          wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
        else:
          ones  = wqb_list[k].eq(1.)
          wq_list[ones] = -wq_list[ones]
    return wq_list


def int_to_twosComplement(weight_q, w_bitwidth: int=8, device='cpu'):
    weight_q = weight_q.to(device, copy=True)
    is_neg = weight_q.lt(0.)
    weight_q[is_neg] = 2**(w_bitwidth-1) + weight_q[is_neg]

    weight_q_shape = list(weight_q.size())
    remainder_list = torch.zeros([w_bitwidth] + weight_q_shape, device=device)
    for k in reversed(range(w_bitwidth)):
        remainder = torch.fmod(weight_q, 2)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/2)
    
    remainder_list[0, is_neg] = 1.
    return remainder_list


def twosComplement_to_int(wqb_list, w_bitwidth: int=8, device='cpu'):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape, device=device)

    for k in reversed(range(int(w_bitwidth))):
        if k != 0:
          wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
        else:
          wq_list -= (wqb_list[k] * 2.**(w_bitwidth-1-k))

    return wq_list


def int_to_binary(weight_q, w_bitwidth: int=8, device='cpu'):
    weight_q = weight_q.clone()
    
    weight_q_shape = list(weight_q.size())
    remainder_list = torch.zeros([w_bitwidth] + weight_q_shape, device=device)
    for k in reversed(range(w_bitwidth)):
        remainder = torch.fmod(weight_q, 2)
        remainder_list[k] = remainder
        weight_q = torch.round((weight_q-remainder)/2)
    return remainder_list


def binary_to_int(wqb_list, w_bitwidth=8, device='cpu'):
    bin_list_shape = wqb_list.size()
    wq_list_shape = list(bin_list_shape[1:])
    wq_list = torch.zeros(wq_list_shape, device=device)
    for k in reversed(range(int(w_bitwidth))):
        wq_list += (wqb_list[k] * 2.**(w_bitwidth-1-k))
    return wq_list


def take_twosComplement(wqb_list, w_bitwidth=8, cellbit=1):
    '''
    Take 2's complement of a number. 
    E.g.,   81  = b'01010001
    return, -81 = b'10101111
    '''
    wqb_list = wqb_list.clone()
    new_wqb_list = torch.zeros_like(wqb_list)

    ones  = wqb_list.eq(1.)
    zeros = wqb_list.eq(0.)

    # invert the bits for adding 1
    wqb_list[ones]  = 0.
    wqb_list[zeros] = 1.

    ones  = wqb_list.eq(1.)
    zeros = wqb_list.eq(0.)

    for k in range(int(w_bitwidth/cellbit)):
        wqb    = wqb_list[k]
        is_one = ones[k]
        is_zrs = zeros[k]
        
        if k == 0:
            wqb[is_one]   = 0.
            wqb[is_zrs]   = 1.
            propagate_one = is_one
        else:
            wqb[is_one*propagate_one] = 0.
            wqb[is_zrs*propagate_one] = 1.
            propagate_one = is_one * propagate_one
        
        new_wqb_list[w_bitwidth - k] = wqb
    return new_wqb_list


def inv_less_bit_twosComplement_conv(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C, H, W = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    wqb_twosComplement[skip_zero, :] = 1 - wqb_twosComplement[skip_zero, :]

    return wqb_twosComplement


def inv_less_bit_twosComplement_fc(wq_int, w_bitwidth=8, group_size=16, device='cpu'):
    K, C = wq_int.shape # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)
    bit_one_count = torch.sum(wqb_twosComplement, dim=-1)
    skip_zero = bit_one_count.lt(group_size/2)
    wqb_twosComplement[skip_zero, :] = 1 - wqb_twosComplement[skip_zero, :]

    return wqb_twosComplement