"""A WrenNet checkpoint with another number of classes, to test that the pipeline is not tied to the bird model's 9:
the bird checkpoint with its classifier cut to the first N rows (its predictions mean nothing).
Usage: python tests/make_test_checkpoint.py IN.pt OUT.pt N"""
import sys
import torch

sd = torch.load(sys.argv[1], map_location="cpu")
n = int(sys.argv[3])
sd["fc.weight"], sd["fc.bias"] = sd["fc.weight"][:n].clone(), sd["fc.bias"][:n].clone()
torch.save(sd, sys.argv[2])
print(f"{sys.argv[2]}: {n} classes")
