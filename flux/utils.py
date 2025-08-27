import torch
import textwrap
import os 

def select_gpu(id_string)
    control_vars=["CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
    for a in control_vars:
        os.environ(a) = str(id_string)

def checkseed(seed):
    mi = -pow(2, 63) 
    ma = pow(2, 63) -1 
    return mi <= seed <= ma

def restate(seed):

    he = '%X' % seed
    he = he.rjust(32, "0")

    re = reversed(textwrap.wrap(he, 2))
    lt = []

    for a in re:
        lt.append(int(a,16))
    
    rt = torch.tensor(lt, dtype=torch.uint8)
    return rt

def state_to_seed_hex(state):
    c = "0x"
    for a in reversed(state):
        b = '%X' % a
        if (len(b) < 2):
            b = f"0{b}"
        c = c + b

    return c 

def state_to_seed(state):
    c = state_to_seed_hex(state)
    return int(c,16)

def report_state(state):
    h = state_to_seed_hex(state)
    #i = state_to_seed(state) # this breaks due to the size of the state on CPU
    #print(f"State: torch.{state} || {h} || {i}")
    print(f"State: torch.{state} || {h}")

def init_rng(platform, seed=None):
    if platform["name"] == "Habana":
        print("Initialising Habana RNG")
        import habana_frameworks.torch.hpu.random as htrandom
        import habana_frameworks.torch.core as htcore
        generator = htrandom.manual_seed(1234)
    else:
        generator = torch.Generator(platform["device"])
    if seed != None:
        if type(seed) is torch.Tensor:
            print(f"Recovering generator state to: {seed}")
            generator.set_state(seed)
        else: 
            print(f"Setting seed to {seed}")
            if checkseed(seed):
                generator.manual_seed(seed)
            else:
                print(f"Seed too long to use .seed - converting to tensor.")
                tseed = restate(seed)
                print(f"Converted tensor: {tseed}")
                generator.set_state(tseed)
    else:
        print("No seed.")
        generator.seed()

    return generator