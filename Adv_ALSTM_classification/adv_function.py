import torch


def get_adv(origin_input,label,final_map,model,epsilon,loss_function):
    '''
    origin_input: batch_size * T * feature_dim
    y_label: batch_size
    final_map: final mapping layer
    model: Attentive lSTM
    epsilon: learning rate to control the adv examples
    criterion: loss function
    '''
    e_s = model(origin_input)
    e_s.retain_grad()
    y_s = final_map(e_s)
    loss_1 = loss_function(y_s, label.flatten())
    g_s = torch.autograd.grad(outputs = loss_1,inputs=e_s,grad_outputs=None)[0]
    g_snorm = torch.sqrt(torch.norm(g_s,p = 2))
    if g_snorm == 0:
        return 0
    else:
        r_adv = epsilon*(g_s/g_snorm)
        return r_adv.detach()