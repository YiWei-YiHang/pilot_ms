
def revise_pilot_unet_cross_attention_forward(unet):
    pass
def revise_pilot_unet_self_attention_forward(unet):
    pass
def revise_pilot_unet_attention_forward(unet):
    revise_pilot_unet_cross_attention_forward(unet)
    revise_pilot_unet_self_attention_forward(unet)