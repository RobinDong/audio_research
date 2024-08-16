import torch
import timm

EMBED_DIM = 768


class PatchEmbed(torch.nn.Module):
    def __init__(self, old, fstride, tstride):
        super().__init__()
        self.proj = torch.nn.Conv2d(
            1, EMBED_DIM, kernel_size=(16, 16), stride=(fstride, tstride)
        )
        self.proj.weight = torch.nn.Parameter(
            torch.sum(old.patch_embed.proj.weight, dim=1).unsqueeze(1)
        )
        self.proj.bias = torch.nn.Parameter(old.patch_embed.proj.bias)

    def forward(self, inp):
        return self.proj(inp).flatten(2).transpose(1, 2)


class MyTranspose(torch.nn.Module):
    def forward(self, x):
        return torch.transpose(x, 1, 2)


def get_shape(fstride, tstride, input_fdim, input_tdim):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = torch.nn.Conv2d(
        1, EMBED_DIM, kernel_size=(16, 16), stride=(fstride, tstride)
    )
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return f_dim, t_dim


def transfer_model(model, fstride=10, tstride=10, input_fdim=128, input_tdim=998):
    # model.patch_embed = PatchEmbed(model, fstride, tstride)
    drop_rate = 0.0
    pre_model = timm.create_model(
        "convnext_small_in22ft1k",
        # img_size=(MELS, 998),
        in_chans=1,
        drop_rate=drop_rate,
        drop_path_rate=drop_rate,
        pretrained=True,
    )
    # Before Pool: [16, 768, 4, 31]
    # After  Pool: [16, 768, 1, 1]
    layers = list(pre_model.children())[:-2]
    layers.append(torch.nn.Flatten(start_dim=2))
    layers.append(MyTranspose())
    model.patch_embed = torch.nn.Sequential(*layers)

    original_num_patches = 576
    oringal_hw = 384 // 16
    # f_dim, t_dim = get_shape(fstride, tstride, input_fdim, input_tdim)
    f_dim, t_dim = 4, 31
    num_patches = f_dim * t_dim

    new_pos_embed = (
        model.pos_embed[:, 2:, :]
        .detach()
        .reshape(1, original_num_patches, EMBED_DIM)
        .transpose(1, 2)
        .reshape(1, EMBED_DIM, oringal_hw, oringal_hw)
    )
    # cut (from middle) or interpolate the second dimension of the positional embedding
    if t_dim <= oringal_hw:
        new_pos_embed = new_pos_embed[
            :,
            :,
            :,
            int(oringal_hw / 2)
            - int(t_dim / 2) : int(oringal_hw / 2)
            - int(t_dim / 2)
            + t_dim,
        ]
    else:
        new_pos_embed = torch.nn.functional.interpolate(
            new_pos_embed, size=(oringal_hw, t_dim), mode="bilinear"
        )
    # cut (from middle) or interpolate the first dimension of the positional embedding
    if f_dim <= oringal_hw:
        new_pos_embed = new_pos_embed[
            :,
            :,
            int(oringal_hw / 2)
            - int(f_dim / 2) : int(oringal_hw / 2)
            - int(f_dim / 2)
            + f_dim,
            :,
        ]
    else:
        new_pos_embed = torch.nn.functional.interpolate(
            new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
        )
    # flatten the positional embedding
    new_pos_embed = new_pos_embed.reshape(1, EMBED_DIM, num_patches).transpose(1, 2)
    # concatenate the above positional embedding with
    # the cls token and distillation token of the deit model.
    model.pos_embed = torch.nn.Parameter(
        torch.cat([model.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
    )
