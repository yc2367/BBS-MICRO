import torch
import torch.nn as nn
import torch.nn.functional as F
from bin_int_convert import *


def bitFlip_conv(wq_int, w_bitwidth: int=8, group_size: int=16, num_pruned_column: int=4, device='cpu'):
    wq_int = wq_int.to(device)
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

    # check existing zero columns
    zero_column_mask = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(w_bitwidth):
        eq_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
        zero_column_mask[i][eq_zero] = True
    
    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([NUM_GROUP], w_bitwidth-num_pruned_column, device=device)
    # cHeck if there are zero columns before prune_until
    for i in range(1, w_bitwidth):
        mask = torch.logical_and(prune_until.gt(i), zero_column_mask[i])
        prune_until[mask] += 1
    
    # test_until is a pointer to specify which column to test until
    test_until = torch.full([NUM_GROUP], 1, device=device)
    for i in range(1, w_bitwidth):
        mask = torch.logical_and(prune_until.gt(i), zero_column_mask[i])
        test_until[mask] = i + 1
    
    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    value_test = torch.zeros_like(wq_int, dtype=torch.float32, device=device)
    value_new = torch.zeros_like(wq_int, dtype=torch.float32, device=device)

    for test_idx in range(1, w_bitwidth):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test
    
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group = mask_group.unsqueeze(-1).expand(-1, group_size)
            error = torch.full([NUM_GROUP, group_size], 1e7, device=device)
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group] = column_new[:, mask_group]
    
    wq_int_new = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, H, W, C).permute(0, 3, 1, 2)

    return wq_int_new


def bitFlip_fc(wq_int, w_bitwidth: int=8, group_size: int=16, num_pruned_column: int=4, device='cpu'):
    wq_int = wq_int.to(device)
    K, C = wq_int.size() # output channel, input channel
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_signMagnitude = int_to_signMagnitude(wq_int, w_bitwidth=w_bitwidth, device=device)

    # check existing zero columns
    zero_column_mask = torch.zeros([w_bitwidth, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(w_bitwidth):
        eq_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
        zero_column_mask[i][eq_zero] = True

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([NUM_GROUP], w_bitwidth-num_pruned_column, device=device)
    # cHeck if there are zero columns before prune_until
    for i in range(1, w_bitwidth):
        mask = torch.logical_and(prune_until.gt(i), zero_column_mask[i])
        prune_until[mask] += 1
    
    # test_until is a pointer to specify which column to test until
    test_until = torch.full([NUM_GROUP], 1, device=device)
    for i in range(1, w_bitwidth):
        mask = torch.logical_and(prune_until.gt(i), zero_column_mask[i])
        test_until[mask] = i + 1
    
    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    value_test = torch.zeros_like(wq_int, dtype=torch.float32, device=device)
    value_new = torch.zeros_like(wq_int, dtype=torch.float32, device=device)

    for test_idx in range(1, w_bitwidth):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test
    
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group = mask_group.unsqueeze(-1).expand(-1, group_size)
            error = torch.full([NUM_GROUP, group_size], 1e7, device=device)
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group] = column_new[:, mask_group]
    
    wq_int_new = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, C)

    return wq_int_new

