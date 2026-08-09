import re, sys, shutil, time, os
p = os.path.expanduser("~/lingbot-map/demo_render/demo.py")
bak = p + ".bak_lowres"
if not os.path.exists(bak):
    shutil.copy(p, bak)
    print("backup:", bak)
else:
    print("backup already exists:", bak)
src = open(p).read()

if "_adapt_pos_embed" in src:
    print("ALREADY PATCHED"); sys.exit(0)

helper = '''
def _resize_pos_embed_tensor(pe, target_len, num_prefix=1):
    import math
    import torch.nn.functional as F
    dim = pe.shape[-1]
    prefix = pe[:, :num_prefix]
    patch = pe[:, num_prefix:]
    Nsrc = patch.shape[1]; Msrc = int(round(math.sqrt(Nsrc)))
    Ntgt = target_len - num_prefix; Mtgt = int(round(math.sqrt(Ntgt)))
    patch = patch.reshape(1, Msrc, Msrc, dim).permute(0, 3, 1, 2).float()
    patch = F.interpolate(patch, size=(Mtgt, Mtgt), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, Mtgt * Mtgt, dim).to(pe.dtype)
    return torch.cat([prefix, patch], dim=1)


def _adapt_pos_embed(state_dict, model):
    """Interpolate any pretrained pos_embed whose grid != the (lower-res) model's, so
    --image_size below the native 518 loads cleanly. Standard ViT-at-new-resolution trick."""
    import math
    msd = dict(model.named_parameters())
    for k, v in list(state_dict.items()):
        if not k.endswith("pos_embed"):
            continue
        if k not in msd or msd[k].shape == v.shape:
            continue
        if v.dim() != 3 or msd[k].dim() != 3 or v.shape[-1] != msd[k].shape[-1]:
            continue
        src_len, tgt_len = v.shape[1], msd[k].shape[1]
        prefix = None
        for pfx in (1, 0):
            s = int(round(math.sqrt(src_len - pfx))); t = int(round(math.sqrt(tgt_len - pfx)))
            if s * s == src_len - pfx and t * t == tgt_len - pfx:
                prefix = pfx; break
        if prefix is None:
            print(f"  [lowres] skip {k}: non-square grid {src_len}->{tgt_len}")
            continue
        state_dict[k] = _resize_pos_embed_tensor(v, tgt_len, prefix)
        print(f"  [lowres] interpolated {k}: {tuple(v.shape)} -> {tuple(state_dict[k].shape)}")
    return state_dict


'''

# 1) insert helpers right before "def load_model("
anchor_fn = "def load_model(args, device):"
assert anchor_fn in src, "load_model anchor not found"
src = src.replace(anchor_fn, helper + anchor_fn, 1)

# 2) call it right after state_dict is obtained, before load_state_dict
anchor_load = '        state_dict = ckpt.get("model", ckpt)\n        missing, unexpected = model.load_state_dict(state_dict, strict=False)'
assert anchor_load in src, "load_state_dict anchor not found"
src = src.replace(anchor_load,
    '        state_dict = ckpt.get("model", ckpt)\n'
    '        state_dict = _adapt_pos_embed(state_dict, model)  # lower-res pos_embed support\n'
    '        missing, unexpected = model.load_state_dict(state_dict, strict=False)', 1)

open(p, "w").write(src)
print("PATCHED OK")
