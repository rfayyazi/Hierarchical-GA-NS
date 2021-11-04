

if __name__ == "__main__":
    for theta in policy.parameters():
        theta.requires_grad = False