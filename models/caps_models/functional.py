import torch
import torch.nn.functional as F


def squash(x, dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 + norm_squared)
    part2 = x / torch.sqrt(norm_squared + 1e-16)
    output = part1 * part2
    return output


def cosine_similarity(x, eps=1e-8):
    """
    Args:
        x (N, out_capsules, 2b, 2b, 2b, input_capsules, output_capsule_dim)
    Return:
        result: (N, out_capsules, 2b, 2b, 2b, input_capsules, input_capsules)
    """
    numerator = torch.matmul(x, x.transpose(-1, -2))

    # compute the length of each vector (for each row)
    # 1. remove square, use L2 norm
    denominator = torch.norm(x, dim=-1, keepdim=True)
    # 2. avoid dividing by zero
    eps_matrix = eps * torch.ones_like(denominator)
    denominator = torch.max(denominator, eps_matrix)
    # 3. compute results multiplied by two vector length
    denominator = denominator * denominator.transpose(-1, -2)

    return numerator / denominator


def degree_score(x, eps=1e-8):
    """
    Args:
        x  u_hat (N, input_capsules, out_capsules, output_capsule_dim, 2b, 2b, 2b )
    Return:
        result   (N, input_capsules, out_capsules,          1,         2b, 2b, 2b )
    """
    x = x.permute(0, 2, 4, 5, 6, 1, 3)
    cosine_score = cosine_similarity(x, eps)

    result = F.softmax(cosine_score.sum(dim=-1, keepdim=True), dim=-1)
    result = result.permute(0, 5, 1, 6, 2, 3, 4)
    return result
