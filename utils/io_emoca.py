## Help functions for EMOCA's encoders
## Copied and refactored from gdl/models/DECA.py and gdl_apps/EMOCA/utils/io.py
import torch

def custom_encode_values(img, coarse_shape_encoder, expression_encoder, detail_encoder):
    img["image"] = img["image"].cuda()
    if len(img["image"].shape) == 3:
        img["image"] = img["image"].view(1,3,224,224)
    vals = custom_encode(img["image"], coarse_shape_encoder, expression_encoder, detail_encoder)
    vals['posecode'][0,:3] = 0
    vals['shapecode'][[0]] = 0
    vals['lightcode'][[0]] = 0
    vals['cam'][0,0] = vals['cam'][0,0] * 0.75
    return vals

def custom_encode(image, coarse_shape_encoder, expression_encoder, detail_encoder):
    codedict = {}

    if len(image.shape) == 5:
            K = image.shape[1]
    elif len(image.shape) == 4:
        K = 1
    else:
        raise RuntimeError("Invalid image batch dimensions.")

    # [B, K, 3, size, size] ==> [BxK, 3, size, size]
    image = image.view(-1, image.shape[-3], image.shape[-2], image.shape[-1])
    code, original_code = emoca_encode_flame(image, coarse_shape_encoder=coarse_shape_encoder, expression_encoder=expression_encoder)
    shapecode, texcode, expcode, posecode, cam, lightcode = emoca_unwrap_list(code)
    if original_code is not None:
        original_code = emoca_unwrap_list_to_dict(original_code)
    
    all_detailcode = detail_encoder(image)

    detailcode = all_detailcode[:, :128]
    detailemocode = all_detailcode[:, 128:(128 + 0)]

    codedict['shapecode'] = shapecode
    codedict['texcode'] = texcode
    codedict['expcode'] = expcode
    codedict['posecode'] = posecode
    codedict['cam'] = cam
    codedict['lightcode'] = lightcode
    codedict['detailcode'] = detailcode
    codedict['detailemocode'] = detailemocode
    codedict['images'] = image

    if original_code is not None:
        codedict['original_code'] = original_code

    return codedict

# EMOCA help functions
def emoca_encode_flame(image, coarse_shape_encoder, expression_encoder):
    deca_code, _ = deca_encode_flame(image, coarse_shape_encoder)
    emoca_code = expression_encoder(image)
    codelist, original_code = emoca_decompose_code((deca_code, emoca_code))
    return codelist, original_code

def emoca_encode_flame_no_decompose(image, coarse_shape_encoder, expression_encoder):
    deca_code, _ = deca_encode_flame(image, coarse_shape_encoder)
    emoca_code = expression_encoder(image)
    return deca_code, emoca_code

def emoca_decompose_code(code):
    deca_code_list = code[0]
    expdeca_code = code[1]
    # deca_code_list, _ = deca_decompose_code(deca_code)
    exp_idx = 2
    deca_code_list_copy = deca_code_list.copy()
    deca_code_list[exp_idx] = expdeca_code

    return deca_code_list, deca_code_list_copy

def emoca_unwrap_list(codelist):
    shapecode, texcode, expcode, posecode, cam, lightcode = codelist
    return shapecode, texcode, expcode, posecode, cam, lightcode

def emoca_unwrap_list_to_dict(codelist): 
    shapecode, texcode, expcode, posecode, cam, lightcode = codelist
    return {'shape': shapecode, 'tex': texcode, 'exp': expcode, 'pose': posecode, 'cam': cam, 'light': lightcode}

# DECA help functions
def deca_encode_flame(image, encoder):
    with torch.no_grad():
        parameters = encoder(image)
    code_list, original_code = deca_decompose_code(parameters)
    return code_list, original_code

def deca_decompose_code(code):
    code_list = []
    # Numerical values copied directly from EMOCA_v2_lr_mse_20's config
    # num_list = [self.config.n_shape, self.config.n_tex, self.config.n_exp, self.config.n_pose, self.config.n_cam, self.config.n_light]
    num_list = [100, 50, 50, 6, 3, 27]
    start = 0
    for i in range(len(num_list)):
        code_list.append(code[:, start:start + num_list[i]])
        start = start + num_list[i]
    code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
    return code_list, None
