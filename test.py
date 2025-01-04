import torch
from vit_pytorch import ViT

def test():
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 10,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    
    probs = torch.nn.functional.softmax(preds, dim = -1)

    print(probs)
    # It shows tensor([[ 0.1093, -0.2483,  0.0897,  0.2005,  0.1606,  0.7256, -0.4171, -0.7569,
     #    -0.3613, -1.3406]], grad_fn=<AddmmBackward0>)
    
    
    # The output is a tensor of shape (1, 10), which is the number of classes in the model.

    # now you can train it as any other pytorch model


if __name__ == '__main__':
    test()