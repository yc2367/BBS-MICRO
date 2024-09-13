import torch
import torch.nn as nn
import torch.nn.functional as F
from bin_int_convert import *


def roundAvg_conv(wq_int, w_bitwidth: int=8, group_size: int=16, num_pruned_column: int=4, device='cpu'):
    wq_int = wq_int.to(device)
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([NUM_GROUP], w_bitwidth-num_pruned_column, device=device)
    eq_msb_column = torch.ones([NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(1, w_bitwidth-4):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        prune_until[eq_msb_column] += 1
    
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        mask_group = torch.eq(prune_until, prune_idx)
        mask_value = mask_group.unsqueeze(-1).expand(-1, group_size)
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        value_mean = torch.round(torch.mean(value_test, dim=-1))
        value_mean = value_mean.unsqueeze(-1).expand(-1, group_size)
        column_new = int_to_binary(value_mean, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]

    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, H, W, C).permute(0, 3, 1, 2)

    return wq_int_new


def roundAvg_fc(wq_int, w_bitwidth: int=8, group_size: int=16, num_pruned_column: int=4, device='cpu'):
    wq_int = wq_int.to(device)
    K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.view(NUM_GROUP, group_size)

    wqb_twosComplement = int_to_twosComplement(wq_int, w_bitwidth=w_bitwidth, device=device)

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([NUM_GROUP], w_bitwidth-num_pruned_column, device=device)
    eq_msb_column = torch.ones([NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(1, w_bitwidth-4):
        eq_column = torch.all(torch.eq(wqb_twosComplement[0], wqb_twosComplement[i]), dim=-1)
        eq_msb_column = torch.logical_and(eq_msb_column, eq_column)
        prune_until[eq_msb_column] += 1
    
    column_test = torch.zeros_like(wqb_twosComplement, dtype=torch.float32, device=device)
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        mask_group = torch.eq(prune_until, prune_idx)
        mask_value = mask_group.unsqueeze(-1).expand(-1, group_size)
        column_test[prune_idx:, mask_value] = wqb_twosComplement[prune_idx:, mask_value]
        value_test = binary_to_int(column_test[prune_idx:], w_bitwidth=w_bitwidth-prune_idx, device=device)
        value_mean = torch.round(torch.mean(value_test, dim=-1))
        value_mean = value_mean.unsqueeze(-1).expand(-1, group_size)
        column_new = int_to_binary(value_mean, w_bitwidth=w_bitwidth-prune_idx, device=device)
        wqb_twosComplement[prune_idx:, mask_value] = column_new[:, mask_value]

    wq_int_new = twosComplement_to_int(wqb_twosComplement, w_bitwidth=w_bitwidth, device=device)
    wq_int_new = wq_int_new.view(K, C)

    return wq_int_new


def zeroPointShifting_conv(wq_int, w_bitwidth: int=8, group_size: int=16, 
                           num_pruned_column: int=4, const_bitwidth: int=5, device='cpu'):
    wq_int = wq_int.to(device)
    K, C, H, W = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*W*H*C//group_size
    wq_int = wq_int.permute([0, 2, 3, 1]).unsqueeze(-1)
    wq_int = wq_int.view(NUM_GROUP, group_size)

    # clipping threshold
    v_max = 2.**(w_bitwidth-1) - 1
    v_min = -v_max
    offset_min = -2**int(const_bitwidth-1)
    offset_max = 2**int(const_bitwidth-1)
    rp_factor = offset_max - offset_min

    wq_int_rp = wq_int.unsqueeze(0).repeat(rp_factor, 1, 1)
    for i, offset in enumerate(range(offset_min, offset_max)):
        wq_int_rp[i] = wq_int_rp[i] + float(offset)
    
    wq_int_rp[wq_int_rp.lt(v_min)] = v_min
    wq_int_rp[wq_int_rp.gt(v_max)] = v_max

    wqb_signMagnitude = int_to_signMagnitude(wq_int_rp, w_bitwidth=w_bitwidth, device=device)

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([rp_factor, NUM_GROUP], int(w_bitwidth-num_pruned_column), device=device)
    # test_until is a pointer to specify which column to test until
    test_until  = torch.full([rp_factor, NUM_GROUP], 1, device=device)
    # Boolean mask to indicate zero MSB column
    is_msb_zero = torch.ones([rp_factor, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(1, w_bitwidth):
        is_current_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
        is_msb_zero = torch.logical_and(is_msb_zero, is_current_zero)
        prune_until[is_msb_zero] +=  1
        test_until[is_msb_zero]  +=  1

    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    value_test = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)
    value_new = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)

    for test_idx in range(1, w_bitwidth):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test
    
    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group = mask_group.unsqueeze(-1).expand(-1, -1, group_size)
            error = torch.full([rp_factor, NUM_GROUP, group_size], 1e7, device=device)
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group] = column_new[:, mask_group]

    wq_int_pruned = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    for i, offset in enumerate(range(offset_min, offset_max)):
        wq_int_pruned[i] = wq_int_pruned[i] - float(offset)
    
    wq_int_original = wq_int.to(torch.float32)
    wq_int_new = torch.zeros_like(wq_int_original, dtype=torch.float32, device=device)
    error = torch.full([NUM_GROUP], 1e7, device=device)
    for i in range(rp_factor):
        new_error = torch.sum((wq_int_pruned[i] - wq_int_original)**2, dim=-1)
        mask_value = torch.lt(new_error, error)
        error[mask_value] = new_error[mask_value]
        wq_int_new[mask_value] = wq_int_pruned[i][mask_value]

    wq_int_new = wq_int_new.view(K, H, W, C).permute(0, 3, 1, 2)

    return wq_int_new


def zeroPointShifting_fc(wq_int, w_bitwidth: int=8, group_size: int=16, 
                         num_pruned_column: int=4, const_bitwidth: int=5, device='cpu'):
    wq_int = wq_int.to(device)
    K, C = wq_int.size() # output channel, input channel, kernel width, kernel height
    if C < group_size:
        group_size = C
    NUM_GROUP = K*C//group_size
    wq_int = wq_int.view(NUM_GROUP, group_size)

    # clipping threshold
    v_max = 2.**(w_bitwidth-1) - 1
    v_min = -v_max
    offset_min = -2**int(const_bitwidth-1)
    offset_max = 2**int(const_bitwidth-1)
    rp_factor = offset_max - offset_min
    
    wq_int_rp = wq_int.unsqueeze(0).repeat(rp_factor, 1, 1)
    for i, offset in enumerate(range(offset_min, offset_max)):
        wq_int_rp[i] = wq_int_rp[i] + float(offset)
    
    wq_int_rp[wq_int_rp.lt(v_min)] = v_min
    wq_int_rp[wq_int_rp.gt(v_max)] = v_max

    wqb_signMagnitude = int_to_signMagnitude(wq_int_rp, w_bitwidth=w_bitwidth, device=device)

    # prune_until is a pointer to specify which column to prune until
    # E.g., for 8-bit weight -> column_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    # if require 4 zero columns, then we should prune from column 7 until column 4 
    prune_until = torch.full([rp_factor, NUM_GROUP], w_bitwidth-num_pruned_column, device=device)
    # test_until is a pointer to specify which column to test until
    test_until  = torch.full([rp_factor, NUM_GROUP], 1, device=device)
    # Boolean mask to indicate zero MSB column
    is_msb_zero = torch.ones([rp_factor, NUM_GROUP], dtype=torch.bool, device=device)
    for i in range(1, w_bitwidth):
        is_current_zero = torch.all(torch.eq(wqb_signMagnitude[i], 0.), dim=-1)
        is_msb_zero = torch.logical_and(is_msb_zero, is_current_zero)
        prune_until[is_msb_zero] +=  1
        test_until[is_msb_zero]  +=  1

    # prune columns [prune_until:], we should test columns [test_until:] to minimize MSE
    # since the columns between test_until and prune_until can be adjusted arbitrarily as long as [prune_until:] are all zero
    value_test = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)
    value_new = torch.zeros_like(wq_int_rp, dtype=torch.float32, device=device)

    for test_idx in range(1, w_bitwidth):
        mask = torch.eq(test_until, test_idx)
        column_test = wqb_signMagnitude[test_idx:, mask, :]
        int_test = binary_to_int(column_test, w_bitwidth=w_bitwidth-test_idx, device=device)
        value_test[mask] = int_test

    for prune_idx in range(w_bitwidth-num_pruned_column, w_bitwidth):
        for test_idx in range(1, prune_idx):
            mask_group = torch.logical_and(torch.eq(test_until, test_idx), torch.eq(prune_until, prune_idx))
            mask_group = mask_group.unsqueeze(-1).expand(-1, -1, group_size)
            error = torch.full([rp_factor, NUM_GROUP, group_size], 1e7, device=device)
            for n in range(2**(prune_idx-test_idx)):
                tmp_value = n * 2.**(w_bitwidth-prune_idx)
                new_error = (tmp_value - value_test) ** 2
                mask_value = torch.logical_and(torch.lt(new_error, error), mask_group)
                error[mask_value] = new_error[mask_value]
                value_new[mask_value] = tmp_value
            column_new = int_to_binary(value_new, w_bitwidth=w_bitwidth-test_idx, device=device)
            wqb_signMagnitude[test_idx:, mask_group] = column_new[:, mask_group]
    
    wq_int_pruned = signMagnitude_to_int(wqb_signMagnitude, w_bitwidth=w_bitwidth, device=device)
    for i, offset in enumerate(range(offset_min, offset_max)):
        wq_int_pruned[i] = wq_int_pruned[i] - float(offset)
    
    wq_int_original = wq_int.to(torch.float32)
    wq_int_new = torch.zeros_like(wq_int_original, dtype=torch.float32, device=device)
    error = torch.full([NUM_GROUP], 1e7, device=device)
    for i in range(rp_factor):
        new_error = torch.sum((wq_int_pruned[i] - wq_int_original)**2, dim=-1)
        mask_value = torch.lt(new_error, error)
        error[mask_value] = new_error[mask_value]
        wq_int_new[mask_value] = wq_int_pruned[i][mask_value]

    wq_int_new = wq_int_new.view(K, C)

    return wq_int_new

    