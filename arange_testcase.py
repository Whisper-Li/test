import torch 
import numpy as np 
import sys 
import copy 
from util_test import compare_res 

def generate_data(min, max, dtype):
    input = np.random.uniform(min, max, ()).astype(dtype)

    npu_input = torch.from_numpy(input)

    return npu_input


def cpu_op_exec(input):
    input.to("cpu")
    res = torch.arange(input)
    res = res.numpy()
    return res


def npu_op_exec(input):
    input = input.to("npu")
    res = torch.arange(input)
    res = res.to("cpu")
    res = res.numpy()
    return res


def npu_op_exec_out(input1, input2):
    input1 = input1.to("npu")
    res = input2.to("npu")
    torch.arange(input1, out=res)
    res = res.to("cpu")
    res = res.numpy()
    return res


def test_arange_float16():
    npu_input1 = generate_data(1, 100, np.float16)
    cpu_output = cpu_op_exec(npu_input1)
    npu_output = npu_op_exec(npu_input1)
    compare_res(cpu_output, npu_output, sys._getframe().f_code.co_name)


def test_arange_float16_out():
    npu_input1 = generate_data(1, 100, np.float16)
    npu_input2 = generate_data(1, 100, np.float16)
    cpu_output = cpu_op_exec(npu_input1)
    npu_output = npu_op_exec_out(npu_input1, npu_input2)
    compare_res(cpu_output, npu_output, sys._getframe().f_code.co_name)


def test_arange():
    test_arange_float16()
    test_arange_float16_out()

if __name__ == '__main__': 
    torch.npu.set_device("npu:5")
    test_arange()