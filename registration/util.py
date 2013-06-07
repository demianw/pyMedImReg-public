

def vectorized_dot_product(a, b):
    return (a[..., None] * b[:, None, ...]).sum(-2)
